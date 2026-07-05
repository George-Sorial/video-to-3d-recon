# Design Notes

## Why two pipelines

- `slam/` is the "build it yourself" path: monocular depth + pose + TSDF fusion. Every piece is
  independently inspectable and debuggable, at the cost of scale-ambiguous monocular depth
  needing active handling to keep the fused mesh consistent.
- `gaussian_splatting/` is the "best available framework" path: COLMAP poses + gsplat/Nerfstudio,
  optimized for a visually compelling, high-fidelity result with much less custom glue code.

Running both against the same input video gives a natural comparison for the write-up: explicit
mesh vs. implicit radiance field, hand-built vs. framework pipeline, geometric accuracy vs.
visual fidelity.

## Component choices

### Camera pose estimation
- **COLMAP** (via `pycolmap`) — default, robust reference poses for both pipelines.
- **hloc** — SuperPoint/SuperGlue-based matching, worth trying if COLMAP struggles on
  low-texture walls / repetitive flooring (common in small indoor scenes).
- **GLOMAP** — global SfM mapper, faster than COLMAP's incremental mapper, drop-in replacement.
- **Custom visual odometry** — the hand-coded SLAM piece: feature tracking → essential
  matrix/PnP → local bundle adjustment → loop closure. Evaluated against COLMAP via ATE/RPE.
- **MASt3R-SLAM / DROID-SLAM** — optional learned-SLAM baseline for comparison only, not
  something to hand-code, but useful to know how far off a from-scratch tracker is from SOTA.

### Monocular depth
- **Depth Anything V2** — default. Fast, robust, both relative and metric checkpoints available.
- **Metric3D v2** — alternative if per-frame absolute-scale depth reduces drift enough to justify
  the swap.
- **Depth Anything 3** — a newer joint pose+depth+reconstruction foundation model. Worth a
  side-by-side comparison against the modular pipeline as an ablation, even if not used as the
  primary path.

### Fusion
- **Open3D `ScalableTSDFVolume`** — standard choice for RGB-D-style fusion once pose + depth are
  available per frame.
- Key risk: monocular depth is only correct up to an unknown, frame-varying scale/shift. Mitigate
  by anchoring each frame's depth to the sparse COLMAP point cloud (least-squares scale-and-shift
  fit) before integrating into the TSDF volume, rather than trusting raw network depth.

### Gaussian Splatting
- **Nerfstudio + splatfacto** — highest-level, least glue code, has a built-in web viewer for
  presentation. Recommended default.
- **gsplat** directly — lower-level control, e.g. to initialize from the `slam/` pipeline's point
  cloud instead of COLMAP's.
- **Original Inria 3DGS repo** — reference implementation; useful to read, non-commercial license,
  rougher to integrate.

### Semantics (optional extension)
- 2D segmentation (SAM2 / Grounded-SAM / YOLO-World) fused into 3D using the same poses/depth
  already computed — keeps semantic predictions aligned with geometry by construction, since
  they ride on the same camera model as the reconstruction itself.

## Open questions / things to validate empirically
- How much does scale drift actually show up over a ~30–60s room video before correction?
- Does hloc materially help over vanilla COLMAP for the specific room(s) being captured?
- Is Metric3D v2's absolute depth accurate enough to skip the COLMAP-anchoring step entirely?
