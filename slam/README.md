# SLAM / Monocular Depth + Fusion Pipeline

Video → camera poses (COLMAP) → per-frame metric depth (Depth Anything V2) →
scale-aligned TSDF fusion → a single textured mesh.

## Capture matters most

This pipeline (like all Structure-from-Motion) needs the camera to **translate
through space**, not just rotate. Filming from one spot while panning/turning
is a degenerate case — COLMAP cannot triangulate and the reconstruction
fragments. When capturing:

- **Walk around** the space; sidestep along walls, arc around furniture.
- Keep each object visible from **at least 3 distinct positions**.
- Move **slowly and smoothly** to avoid motion blur.
- 1–2 minutes is plenty for a small room. Good, even lighting; avoid mirrors,
  glass, and large blank walls.

## Install

```bash
# PyTorch with a CUDA build matching your driver (example: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r slam/requirements.txt
```

COLMAP is a separate native install: download the CUDA Windows build from
https://github.com/colmap/colmap/releases and put the folder containing
`COLMAP.bat` on your PATH.

## Run

```bash
python -m scripts.run_slam data/raw_videos/my_room.mp4
```

Output mesh lands in `outputs/meshes/my_room.ply`. Inspect it with:

```bash
python -m scripts.view_mesh outputs/meshes/my_room.ply
```

Useful flags: `--fps`, `--max-width`, `--sharpness`, `--voxel-divisor`
(higher = finer mesh, more memory), `--depth-model`.

## How the pieces fit

| Stage | Module | Notes |
|-------|--------|-------|
| Frames | `preprocessing/extract_frames.py` | ffmpeg sample + safe blur filter (never deletes >25%) |
| Poses | `pose_estimation/colmap_pipeline.py` | CLI wrappers: features, matching, mapper, undistort |
| Model IO | `pose_estimation/colmap_io.py` | parses COLMAP TXT (version-stable) into poses/intrinsics/sparse depths |
| Depth | `depth/depth_estimator.py` | Depth Anything V2 metric-indoor, per-frame metric depth |
| Fusion | `fusion/tsdf_fusion.py` | per-frame scale+shift align to COLMAP, then Open3D TSDF |

## The scale problem, and how it's handled

Monocular depth and COLMAP each have an arbitrary scale. For every frame we fit
a per-frame `(scale, shift)` mapping the network depth onto COLMAP's sparse
depths at the pixels where COLMAP has 3D points, so all depth is expressed in
COLMAP's coordinate frame before integration. The mesh is therefore internally
consistent, though in COLMAP's arbitrary units (not metres) unless you add a
global rescale.
