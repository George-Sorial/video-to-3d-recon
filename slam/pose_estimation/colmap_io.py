"""Parse a COLMAP TXT model (cameras.txt, images.txt, points3D.txt).

We read the TXT format rather than using pycolmap's object API here because
the text format is stable across COLMAP versions, so the fusion step won't
break when pycolmap changes its Python bindings. Everything the fusion needs
is derived here: per-image intrinsics, world->camera poses, and the sparse
3D points each image sees (used to align monocular depth to COLMAP's scale).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Frame:
    image_id: int
    name: str
    camera_id: int
    world_to_cam: np.ndarray            # 4x4
    # Sparse observations: pixel (u, v) and the COLMAP-frame depth (z in camera
    # coordinates) of the 3D point seen there. Used for depth scale alignment.
    sparse_uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    sparse_depth: np.ndarray = field(default_factory=lambda: np.zeros((0,)))


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def _quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def read_cameras(cameras_txt: str) -> dict[int, Intrinsics]:
    """Parse cameras.txt. After undistortion the model is PINHOLE
    (params = fx, fy, cx, cy); SIMPLE_PINHOLE (f, cx, cy) is also handled.
    """
    cameras: dict[int, Intrinsics] = {}
    for line in Path(cameras_txt).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id = int(parts[0])
        model = parts[1]
        width, height = int(parts[2]), int(parts[3])
        params = [float(p) for p in parts[4:]]

        if model == "PINHOLE":
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            # Undistortion should always yield PINHOLE; this is a safety net.
            raise ValueError(
                f"Unexpected camera model '{model}' in {cameras_txt}. "
                "Fusion expects an undistorted PINHOLE model."
            )
        cameras[cam_id] = Intrinsics(width, height, fx, fy, cx, cy)
    return cameras


def read_points3d(points3d_txt: str) -> dict[int, np.ndarray]:
    """Parse points3D.txt -> {point3D_id: xyz (world coords)}."""
    points: dict[int, np.ndarray] = {}
    for line in Path(points3d_txt).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        pid = int(parts[0])
        points[pid] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    return points


def read_images(images_txt: str, points3d: dict[int, np.ndarray]) -> list[Frame]:
    """Parse images.txt. Each image is two lines: a pose line, then a line of
    (x y point3D_id) triplets. We keep only observations with a valid
    point3D_id and precompute their camera-frame depth for scale alignment.
    """
    lines = [
        ln for ln in Path(images_txt).read_text().splitlines()
        if ln and not ln.startswith("#")
    ]

    frames: list[Frame] = []
    for i in range(0, len(lines), 2):
        pose = lines[i].split()
        image_id = int(pose[0])
        qw, qx, qy, qz = (float(pose[1]), float(pose[2]), float(pose[3]), float(pose[4]))
        tx, ty, tz = float(pose[5]), float(pose[6]), float(pose[7])
        camera_id = int(pose[8])
        name = pose[9]

        R = _quat_to_rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        world_to_cam = np.eye(4)
        world_to_cam[:3, :3] = R
        world_to_cam[:3, 3] = t

        # Second line: 2D points as (X Y POINT3D_ID) triplets.
        pts = lines[i + 1].split()
        uv_list, depth_list = [], []
        for j in range(0, len(pts), 3):
            pid = int(pts[j + 2])
            if pid == -1 or pid not in points3d:
                continue
            u, v = float(pts[j]), float(pts[j + 1])
            # Depth = z of the 3D point in this camera's frame.
            xyz_world = points3d[pid]
            xyz_cam = R @ xyz_world + t
            z = xyz_cam[2]
            if z <= 0:
                continue
            uv_list.append((u, v))
            depth_list.append(z)

        frame = Frame(
            image_id=image_id,
            name=name,
            camera_id=camera_id,
            world_to_cam=world_to_cam,
            sparse_uv=np.array(uv_list) if uv_list else np.zeros((0, 2)),
            sparse_depth=np.array(depth_list) if depth_list else np.zeros((0,)),
        )
        frames.append(frame)

    frames.sort(key=lambda f: f.name)
    return frames


def load_model(model_txt_dir: str):
    """Convenience loader: returns (cameras, frames, scene_bbox_diagonal).

    The bbox diagonal of the sparse cloud gives a scale-free handle on the
    scene size, used to auto-set the TSDF voxel resolution (COLMAP's absolute
    scale is arbitrary, so a fixed voxel size in meters is meaningless).
    """
    d = Path(model_txt_dir)
    cameras = read_cameras(str(d / "cameras.txt"))
    points3d = read_points3d(str(d / "points3D.txt"))
    frames = read_images(str(d / "images.txt"), points3d)

    if points3d:
        pts = np.stack(list(points3d.values()))
        bbox_diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    else:
        bbox_diag = 1.0

    return cameras, frames, bbox_diag
