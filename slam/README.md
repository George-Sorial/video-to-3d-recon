# SLAM / Monocular Depth + Fusion Pipeline

The "build it yourself" reconstruction path: per-frame monocular depth + per-frame camera pose,
fused into a single consistent mesh.

## Pipeline stages

1. **`preprocessing/`** — extract frames from the input video at a chosen rate, optionally
   undistort using calibration from COLMAP or a checkerboard calibration, discard blurry/
   near-duplicate frames.
2. **`pose_estimation/`** — get a camera pose for every kept frame. Two tracks to build/compare:
   - Wrapper around **COLMAP** (via `pycolmap`) or **hloc** for robust reference poses.
   - Custom **visual odometry**: ORB/SIFT feature tracking → essential matrix / PnP → local
     bundle adjustment (`g2o` or `gtsam`) → simple loop closure. Evaluate against COLMAP's poses
     using ATE/RPE (`eval/`) to know how well the hand-built tracker is doing.
3. **`depth/`** — per-frame monocular depth via **Depth Anything V2** (default) or **Metric3D v2**
   (if absolute-scale depth turns out to reduce drift enough to be worth the swap).
4. **`fusion/`** — integrate (RGB, depth, pose, intrinsics) per frame into an **Open3D
   `ScalableTSDFVolume`**, extract the final mesh via marching cubes. This is where
   monocular-depth scale drift shows up most — plan for a rescaling/consistency step here
   (e.g. anchoring each frame's depth to the sparse COLMAP point cloud, or a per-frame
   scale-and-shift fit) before integration.
5. **`semantics/`** *(optional)* — 2D segmentation (SAM2 / Grounded-SAM / YOLO-World) per frame,
   fused into 3D by voting labels into the same TSDF voxels used for geometry — keeps semantics
   aligned to the geometry by construction.
6. **`eval/`** — trajectory and reconstruction quality metrics.

## Design tradeoffs

See `../docs/design_notes.md` for the reasoning behind these choices and known failure modes.
