# Video → 3D Reconstruction

Reconstruct a geometrically consistent 3D scene from a short handheld phone video of a small
indoor space (e.g. a room). Two independent reconstruction pipelines live in this repo:

- **`slam/`** — a hand-built, modular pipeline: monocular depth estimation + camera pose
  estimation (classical SfM or custom visual odometry) + TSDF volumetric fusion, producing an
  explicit textured mesh. Built for control and understanding, not speed.
- **`gaussian_splatting/`** — a 3D Gaussian Splatting pipeline built on existing frameworks
  (COLMAP + gsplat/Nerfstudio), producing a photorealistic, freely-navigable radiance field.

Both pipelines share the same input video and, where useful, the same camera pose estimates.

## Status

🚧 Scaffolding stage — folder structure, environment, and design docs are in place; pipeline
code is being built out incrementally. See `docs/design_notes.md` for the current plan and
open questions.

## Repo layout

```
video-to-3d-recon/
├── data/
│   ├── raw_videos/         # input .mp4 files (gitignored, tracked via Git LFS if needed)
│   └── processed/          # extracted frames, per-scene working directories
├── slam/                    # build-it-yourself geometric pipeline
│   ├── preprocessing/       # frame extraction, undistortion, frame selection
│   ├── pose_estimation/     # COLMAP/hloc wrapper OR custom visual odometry
│   ├── depth/               # monocular depth model wrappers (Depth Anything V2 / Metric3D)
│   ├── fusion/              # Open3D TSDF integration -> mesh
│   ├── semantics/           # optional: 2D segmentation -> 3D label fusion
│   └── eval/                # trajectory metrics (ATE/RPE) vs. COLMAP or ground truth
├── gaussian_splatting/
│   ├── preprocessing/       # COLMAP/hloc pose + sparse point init
│   ├── training/            # gsplat / Nerfstudio (splatfacto) training configs
│   └── export/              # .ply / .splat export, render flythroughs
├── scripts/                 # top-level entry points (run_slam.sh, run_gsplat.sh, ...)
├── outputs/
│   ├── meshes/               # exported meshes from slam/
│   ├── splats/               # exported gaussian splat files
│   └── videos/               # rendered flythrough videos for presentation
├── notebooks/                # exploratory notebooks / debugging visualizations
└── docs/                     # design notes, tradeoffs, results write-up
```

## Setup

Requires an NVIDIA GPU with a recent CUDA driver.

```bash
git clone <your-repo-url>.git
cd video-to-3d-recon

# Two separate environments are recommended — the SLAM stack and the Gaussian Splatting
# stack pull in different (and sometimes conflicting) versions of PyTorch/CUDA extras.
conda create -n v3d-slam python=3.10 -y
conda activate v3d-slam
pip install -r slam/requirements.txt

conda create -n v3d-gsplat python=3.10 -y
conda activate v3d-gsplat
pip install -r gaussian_splatting/requirements.txt
```

## Running the pipelines

> Fill in once implemented. Target interface:

```bash
# 1. Drop a video in data/raw_videos/my_room.mp4

# 2. SLAM / mesh pipeline
conda activate v3d-slam
bash scripts/run_slam.sh data/raw_videos/my_room.mp4

# 3. Gaussian Splatting pipeline
conda activate v3d-gsplat
bash scripts/run_gsplat.sh data/raw_videos/my_room.mp4
```

Outputs land in `outputs/meshes/`, `outputs/splats/`, and `outputs/videos/`.

## Example input / output

See `docs/results.md` for example videos, reconstructed meshes/splats, and comparison renders
once available.

## Design notes

See `docs/design_notes.md` for the reasoning behind each library choice and known tradeoffs
(scale drift in monocular depth, loop closure strategy, etc).
