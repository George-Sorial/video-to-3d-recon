"""Fuse per-frame monocular depth + COLMAP poses into a single mesh via TSDF.

The hard part of monocular fusion is scale: the depth network's output and
COLMAP's reconstruction each have their own arbitrary scale, and they must be
reconciled or the fused surface will be inconsistent frame-to-frame. For each
frame we fit a per-frame (scale, shift) that maps the network depth onto
COLMAP's sparse depths at the pixels where COLMAP has 3D points. Every frame's
dense depth is then expressed in COLMAP's coordinate frame, so poses and depth
agree and the TSDF integrates coherently.
"""

from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from slam.pose_estimation.colmap_io import Frame, Intrinsics, load_model


def _fit_scale_shift(
    pred_depth_at_pts: np.ndarray,
    colmap_depth: np.ndarray,
    min_points: int = 20,
) -> tuple[float, float] | None:
    """Least-squares fit of colmap_depth ~= s * pred + t, with one round of
    outlier rejection. Returns (s, t) or None if there aren't enough usable
    correspondences (in which case the frame is skipped rather than fused with
    a bad scale).
    """
    if len(pred_depth_at_pts) < min_points:
        return None

    def solve(pred, target):
        A = np.stack([pred, np.ones_like(pred)], axis=1)
        sol, *_ = np.linalg.lstsq(A, target, rcond=None)
        return sol[0], sol[1]

    s, t = solve(pred_depth_at_pts, colmap_depth)

    # Reject correspondences with large residuals, then refit once.
    resid = np.abs(s * pred_depth_at_pts + t - colmap_depth)
    keep = resid < (resid.mean() + 3.0 * resid.std() + 1e-9)
    if keep.sum() >= min_points:
        s, t = solve(pred_depth_at_pts[keep], colmap_depth[keep])

    if not np.isfinite(s) or not np.isfinite(t) or s <= 0:
        return None
    return float(s), float(t)


def _sample_depth_at(pred_depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest-pixel sampling of a depth map at float pixel coords (u, v)."""
    h, w = pred_depth.shape
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    return pred_depth[v, u]


def fuse(
    undistorted_image_dir: str,
    model_txt_dir: str,
    depth_estimator,
    output_mesh_path: str,
    voxel_divisor: float = 512.0,
) -> str:
    """Run the full fusion. `depth_estimator` is any object with an
    `estimate(image_path) -> HxW float32` method (see depth_estimator.py).

    voxel_divisor controls resolution: voxel_size = scene_diagonal / divisor.
    Higher = finer mesh but more memory/time.
    """
    cameras, frames, bbox_diag = load_model(model_txt_dir)
    image_dir = Path(undistorted_image_dir)

    voxel_size = bbox_diag / voxel_divisor
    sdf_trunc = 5.0 * voxel_size
    depth_trunc = 1.5 * bbox_diag  # discard depth farther than the scene extent
    print(
        f"[fuse] scene diagonal={bbox_diag:.3f} (COLMAP units) | "
        f"voxel={voxel_size:.4f} sdf_trunc={sdf_trunc:.4f}"
    )

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    n_used, n_skipped = 0, 0
    for frame in frames:
        img_path = image_dir / frame.name
        if not img_path.exists():
            n_skipped += 1
            continue

        intr: Intrinsics = cameras[frame.camera_id]

        # 1) Predict metric depth for this frame.
        pred = depth_estimator.estimate(str(img_path))  # HxW metres

        # Depth model runs at the image's own resolution; make sure it matches
        # the intrinsics' expected width/height (undistorter can pad/crop).
        if pred.shape[0] != intr.height or pred.shape[1] != intr.width:
            pred_img = Image.fromarray(pred)
            pred_img = pred_img.resize((intr.width, intr.height), Image.BILINEAR)
            pred = np.asarray(pred_img, dtype=np.float32)

        # 2) Align this frame's depth to COLMAP's scale using the sparse points.
        if len(frame.sparse_depth) == 0:
            n_skipped += 1
            continue
        pred_at_pts = _sample_depth_at(pred, frame.sparse_uv)
        valid = pred_at_pts > 0
        fit = _fit_scale_shift(pred_at_pts[valid], frame.sparse_depth[valid])
        if fit is None:
            n_skipped += 1
            continue
        s, t = fit
        depth_colmap = s * pred + t
        depth_colmap[depth_colmap <= 0] = 0.0  # mark invalid as 0

        # 3) Integrate into the TSDF volume.
        color = o3d.geometry.Image(
            np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        )
        depth = o3d.geometry.Image(depth_colmap.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=1.0,          # depth is already in world units
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy
        )
        # Open3D wants the world->camera extrinsic, which is exactly COLMAP's.
        volume.integrate(rgbd, intrinsic, frame.world_to_cam)
        n_used += 1

    print(f"[fuse] integrated {n_used} frames, skipped {n_skipped}.")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    out = Path(output_mesh_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out), mesh)
    print(f"[fuse] wrote mesh: {out}  ({len(mesh.vertices)} vertices)")
    return str(out)
