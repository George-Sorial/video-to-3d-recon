"""Video -> sampled, sharpness-filtered frames for downstream reconstruction."""

import subprocess
from pathlib import Path

import cv2


def extract_frames(video_path: str, out_dir: str, fps: float = 3.0, max_width: int = 1600) -> None:
    """Sample frames from a video at a fixed rate, resizing to a max width.

    Filenames are strictly sequential (frame_00001.jpg, ...), which COLMAP's
    sequential matcher relies on to infer temporal order. ffmpeg also
    auto-applies the video's rotation metadata, so portrait phone clips come
    out upright.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vf = f"fps={fps},scale={max_width}:-1"
    out_pattern = str(out_dir / "frame_%05d.jpg")

    cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", vf, "-q:v", "2", out_pattern]
    subprocess.run(cmd, check=True)


def laplacian_sharpness(image_path: str) -> float:
    """Higher = sharper. Variance of the Laplacian is a cheap, standard blur
    metric. Its absolute scale depends heavily on scene content/resolution,
    so a threshold that works on one clip may not transfer to another.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def filter_blurry_frames(
    frame_dir: str,
    sharpness_threshold: float | None = None,
    max_fraction_removed: float = 0.25,
) -> None:
    """Drop the blurriest frames, then renumber the survivors so the sequence
    has no gaps.

    Safety first: this function will NEVER remove more than
    `max_fraction_removed` of the frames, no matter what threshold is given.
    That guard exists because an over-aggressive threshold once deleted an
    entire dataset silently — the pipeline then "ran" on zero images.

    - If `sharpness_threshold` is None, we auto-pick: keep everything, but if
      some frames are clearly blurrier than the rest, trim up to
      `max_fraction_removed` of the worst ones.
    - If a threshold IS given, we drop frames below it, but still cap total
      removal at `max_fraction_removed`.
    """
    frame_dir = Path(frame_dir)
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        print("[filter] No frames found — nothing to filter.")
        return

    scored = [(f, laplacian_sharpness(str(f))) for f in frames]
    scores = [s for _, s in scored]
    print(
        f"[filter] {len(frames)} frames | sharpness "
        f"min={min(scores):.1f} median={sorted(scores)[len(scores) // 2]:.1f} "
        f"max={max(scores):.1f}"
    )

    max_removable = int(len(frames) * max_fraction_removed)

    # Rank frames worst-first.
    ranked_worst_first = sorted(scored, key=lambda x: x[1])

    if sharpness_threshold is None:
        to_remove = [f for f, _ in ranked_worst_first[:0]]  # default: remove nothing
    else:
        to_remove = [f for f, s in ranked_worst_first if s < sharpness_threshold]

    # Enforce the cap.
    if len(to_remove) > max_removable:
        print(
            f"[filter] threshold would remove {len(to_remove)} frames "
            f"({len(to_remove) / len(frames):.0%}) — capping at {max_removable} "
            f"to protect the dataset."
        )
        to_remove = to_remove[:max_removable]

    remove_set = {f for f in to_remove}
    for f in remove_set:
        f.unlink()

    kept = [f for f, _ in scored if f not in remove_set]
    kept.sort()
    for i, f in enumerate(kept, start=1):
        target = frame_dir / f"frame_{i:05d}.jpg"
        if f != target:
            f.rename(target)

    print(f"[filter] removed {len(remove_set)}, kept {len(kept)}.")
