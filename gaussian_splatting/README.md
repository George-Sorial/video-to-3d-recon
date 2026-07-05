# 3D Gaussian Splatting Pipeline

The "use the best existing framework" reconstruction path: photorealistic, freely-navigable
radiance field, built on top of COLMAP poses.

## Pipeline stages

1. **`preprocessing/`** — same input video → COLMAP (or reuse the poses computed in `../slam/`)
   to get camera intrinsics/extrinsics + a sparse point cloud for Gaussian initialization.
2. **`training/`** — train the splat. Two options, pick based on how much control you want:
   - **Nerfstudio + `splatfacto`** (`ns-process-data` → `ns-train splatfacto`): least glue code,
     built-in web viewer, handles COLMAP internally.
   - **Raw `gsplat`**: train your own loop against the CUDA rasterizer directly — useful if you
     want to initialize Gaussians from the mesh/point cloud produced by `../slam/fusion/` instead
     of COLMAP's sparse points, to make the two pipelines share geometry.
3. **`export/`** — export trained Gaussians to `.ply`/`.splat`, render camera-path flythroughs
   for the presentation deliverable.

## Design tradeoffs

See `../docs/design_notes.md`.
