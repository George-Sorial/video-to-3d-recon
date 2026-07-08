"""End-to-end SLAM / monocular-fusion pipeline: video .mp4 -> textured mesh.

Run from the repo root:
    python -m scripts.run_slam data/raw_videos/my_room.mp4

Stages:
    1. extract frames from the video
    2. drop the blurriest frames (with a safety cap)
    3. COLMAP feature extraction + sequential matching
    4. COLMAP incremental mapping  -> camera poses + sparse cloud
    5. undistort images -> clean PINHOLE model
    6. Depth Anything V2 metric depth per frame
    7. scale-align depth to COLMAP + TSDF fusion -> mesh.ply
"""

import argparse
import sys
from pathlib import Path

# Make `import slam...` work regardless of where the script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slam.preprocessing.extract_frames import extract_frames, filter_blurry_frames
from slam.pose_estimation.colmap_pipeline import (
    run_feature_extraction,
    run_sequential_matcher,
    run_mapper,
    run_image_undistorter,
    run_model_converter,
    summarize_sparse_model,
    largest_model_dir,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to input .mp4")
    ap.add_argument("--name", default=None, help="scene name (default: video stem)")
    ap.add_argument("--fps", type=float, default=3.0, help="frames per second to sample")
    ap.add_argument("--max-width", type=int, default=1600, help="resize long edge to this")
    ap.add_argument("--sharpness", type=float, default=None,
                    help="blur threshold; default None = keep all but auto-trim worst")
    ap.add_argument("--voxel-divisor", type=float, default=512.0,
                    help="TSDF resolution: voxel = scene_diagonal / divisor")
    ap.add_argument("--depth-model", default=None,
                    help="override Depth Anything V2 checkpoint")
    args = ap.parse_args()

    video = Path(args.video)
    name = args.name or video.stem
    work = Path("data/processed") / name
    image_dir = work / "images"
    database = work / "database.db"
    sparse = work / "sparse"
    dense = work / "dense"                       # undistorted images + model
    dense_txt = dense / "sparse"                 # TXT model lands here too
    mesh_out = Path("outputs/meshes") / f"{name}.ply"

    print("== 1/7 extract frames ==")
    extract_frames(str(video), str(image_dir), fps=args.fps, max_width=args.max_width)

    print("== 2/7 filter blurry frames ==")
    filter_blurry_frames(str(image_dir), sharpness_threshold=args.sharpness)

    print("== 3/7 COLMAP features + matching ==")
    run_feature_extraction(str(database), str(image_dir))
    run_sequential_matcher(str(database), loop_detection=False)

    print("== 4/7 COLMAP mapper ==")
    run_mapper(str(database), str(image_dir), str(sparse))
    summarize_sparse_model(str(sparse))
    best_model = largest_model_dir(str(sparse))
    print(f"    using largest model: {best_model}")

    print("== 5/7 undistort ==")
    run_image_undistorter(str(image_dir), best_model, str(dense))
    # image_undistorter writes dense/sparse as .bin; convert it to TXT in place.
    run_model_converter(str(dense / "sparse"), str(dense_txt), output_type="TXT")

    print("== 6/7 + 7/7 depth + fusion ==")
    # Imported here so the earlier COLMAP-only stages don't require torch to be
    # installed just to run steps 1-5.
    from slam.depth.depth_estimator import DepthEstimator, DEFAULT_MODEL
    from slam.fusion.tsdf_fusion import fuse

    estimator = DepthEstimator(model_id=args.depth_model or DEFAULT_MODEL)
    fuse(
        undistorted_image_dir=str(dense / "images"),
        model_txt_dir=str(dense_txt),
        depth_estimator=estimator,
        output_mesh_path=str(mesh_out),
        voxel_divisor=args.voxel_divisor,
    )

    print(f"\nDone. Mesh at: {mesh_out}")
    print("View it with:  python -m scripts.view_mesh", mesh_out)


if __name__ == "__main__":
    main()
