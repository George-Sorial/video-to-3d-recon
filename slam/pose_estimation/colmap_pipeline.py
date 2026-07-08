"""Wrappers around the COLMAP CLI: features, matching, mapping, undistortion."""

import shutil
import subprocess
from pathlib import Path

import pycolmap


def _find_colmap_executable() -> str:
    """COLMAP ships as `colmap` on Linux/Mac and `COLMAP.bat` on Windows —
    detect whichever is actually on PATH. Returns the full resolved path
    (with correct extension), since subprocess.run() with shell=False will
    not auto-append .bat/.cmd itself.
    """
    for candidate in ("colmap", "COLMAP.bat", "colmap.exe"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError(
        "COLMAP executable not found on PATH. Check your PATH entry points "
        "to the folder containing COLMAP.bat / colmap."
    )


COLMAP_CMD = _find_colmap_executable()


def run_feature_extraction(database_path: str, image_dir: str, use_gpu: bool = True) -> None:
    """Extract SIFT features from all images into a COLMAP database.

    Assumes one physical camera for the whole video (fixed intrinsics) and
    the OPENCV camera model so lens distortion is estimated rather than
    baked into the poses.
    """
    cmd = [
        COLMAP_CMD, "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "OPENCV",
        "--FeatureExtraction.use_gpu", "1" if use_gpu else "0",
    ]
    subprocess.run(cmd, check=True)


def run_sequential_matcher(
    database_path: str,
    use_gpu: bool = True,
    loop_detection: bool = False,
    vocab_tree_path: str | None = None,
    overlap: int = 10,
) -> None:
    """Match features between temporally neighboring frames.

    `overlap` sets how many neighbors on each side of a frame get compared —
    raise it if a reconstruction fragments across a rough patch. Loop
    detection is OFF by default because the vocab-tree path crashed on some
    Windows/CUDA COLMAP builds; enable it once you confirm it's stable.
    """
    cmd = [
        COLMAP_CMD, "sequential_matcher",
        "--database_path", database_path,
        "--FeatureMatching.use_gpu", "1" if use_gpu else "0",
        "--SequentialMatching.overlap", str(overlap),
        "--SequentialMatching.loop_detection", "1" if loop_detection else "0",
    ]
    if loop_detection and vocab_tree_path:
        cmd += ["--SequentialMatching.vocab_tree_path", vocab_tree_path]
    subprocess.run(cmd, check=True)


def run_exhaustive_matcher(database_path: str, use_gpu: bool = True) -> None:
    """Match every image against every other. Only practical at small scale
    (tens to low hundreds of images) — handy as a diagnostic or for short
    clips where the sequential neighbor window is too narrow.
    """
    cmd = [
        COLMAP_CMD, "exhaustive_matcher",
        "--database_path", database_path,
        "--FeatureMatching.use_gpu", "1" if use_gpu else "0",
    ]
    subprocess.run(cmd, check=True)


def run_mapper(database_path: str, image_dir: str, output_path: str) -> None:
    """Incremental Structure-from-Motion: turns matched features into camera
    poses + a sparse point cloud. CPU-bound (Ceres bundle adjustment), so it
    will not show GPU activity — that's expected. Writes numbered model
    subfolders under output_path (0/, 1/, ...).
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    cmd = [
        COLMAP_CMD, "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--output_path", output_path,
    ]
    subprocess.run(cmd, check=True)


def run_image_undistorter(
    image_dir: str,
    sparse_model_path: str,
    output_path: str,
    max_image_size: int = 2000,
) -> None:
    """Undistort images and rewrite the model with a plain PINHOLE camera.

    This removes the lens distortion (k1,k2,p1,p2 from the OPENCV model) that
    the downstream TSDF fusion can't account for — after this step the camera
    is a clean pinhole, so depth back-projection is geometrically correct.
    Outputs output_path/images (undistorted) and output_path/sparse (model).
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    cmd = [
        COLMAP_CMD, "image_undistorter",
        "--image_path", image_dir,
        "--input_path", sparse_model_path,
        "--output_path", output_path,
        "--output_type", "COLMAP",
        "--max_image_size", str(max_image_size),
    ]
    subprocess.run(cmd, check=True)


def run_model_converter(input_path: str, output_path: str, output_type: str = "TXT") -> None:
    """Convert a COLMAP model (.bin) to another format. We use TXT because the
    text format is stable across COLMAP versions and trivial to parse, which
    insulates the fusion step from pycolmap API drift.
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    cmd = [
        COLMAP_CMD, "model_converter",
        "--input_path", input_path,
        "--output_path", output_path,
        "--output_type", output_type,
    ]
    subprocess.run(cmd, check=True)


def summarize_sparse_model(output_path: str) -> None:
    """Print registered image + 3D point counts for each model COLMAP made.
    One model containing all images is ideal; several models means some
    frames couldn't be connected (often too little camera translation, or a
    blurry/low-texture patch).
    """
    model_dirs = sorted(p for p in Path(output_path).iterdir() if p.is_dir())
    if not model_dirs:
        print("No models produced — mapper failed to find a good initial pair.")
        return
    for model_dir in model_dirs:
        recon = pycolmap.Reconstruction(str(model_dir))
        print(
            f"{model_dir.name}: {recon.num_reg_images()} images registered, "
            f"{recon.num_points3D()} 3D points"
        )


def largest_model_dir(output_path: str) -> str:
    """Return the model subfolder with the most registered images. COLMAP's
    "0/" is not always the biggest, so pick by image count.
    """
    model_dirs = [p for p in Path(output_path).iterdir() if p.is_dir()]
    if not model_dirs:
        raise RuntimeError(f"No models found under {output_path}.")

    def n_images(model_dir: Path) -> int:
        return pycolmap.Reconstruction(str(model_dir)).num_reg_images()

    best = max(model_dirs, key=n_images)
    return str(best)
