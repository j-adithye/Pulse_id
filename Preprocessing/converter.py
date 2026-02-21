#!/usr/bin/env python3
"""
converter.py  –  Batch preprocessing for the dorsal hand vein dataset.

Walks dataset/ (person1/ … person400/), runs the full preprocessing
pipeline from preprocessing.py on every .jpg image, and saves the
result to processed_dataset/ mirroring the original folder structure.

Usage:
    python converter.py
    python converter.py --dataset dataset --output processed_dataset --workers 4
"""

import os
import sys
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from io import StringIO

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Import every pipeline function directly from preprocessing.py
# (place preprocessing.py in the same directory as this script)
# ---------------------------------------------------------------------------
import preprocessing as pp

# ===========================================================================
#  CONFIGURATION  (mirrors the constants in preprocessing.py)
# ===========================================================================
OTSU_OFFSET          = pp.OTSU_OFFSET
WRIST_JUNCTION_RISE  = pp.WRIST_JUNCTION_RISE
MCP_SAFETY_MARGIN    = pp.MCP_SAFETY_MARGIN
MCP_DEFECT_DEPTH_MIN = pp.MCP_DEFECT_DEPTH_MIN
Sato_SCALE_MIN       = pp.Sato_SCALE_MIN
Sato_SCALE_MAX       = pp.Sato_SCALE_MAX
Sato_SCALE_STEP      = pp.Sato_SCALE_STEP
CLAHE_PRE_CLIP       = pp.CLAHE_PRE_CLIP
CLAHE_POST_CLIP      = pp.CLAHE_POST_CLIP
CLAHE_TILE_SIZE      = pp.CLAHE_TILE_SIZE
TARGET_SIZE          = pp.TARGET_SIZE
# ===========================================================================


def preprocess_single(src_path: Path, dst_path: Path) -> tuple[bool, str]:
    """
    Run the full 11-step pipeline on one image and save the result.

    Returns (success: bool, message: str).
    """
    try:
        with redirect_stdout(StringIO()):  # silence all preprocessing.py prints
            return _run_pipeline(src_path, dst_path)
    except Exception:
        return False, f"FAILED {src_path}:\n{traceback.format_exc()}"


def _run_pipeline(src_path: Path, dst_path: Path) -> tuple[bool, None]:
    try:
        # Step 1 & 2 – Load + Gaussian Blur
        img = pp.load_image(str(src_path))
        img = pp.apply_gaussian_blur(img, kernel_size=11)

        # Step 3 – Smart Crop
        img, _ = pp.smart_crop_hand_region(
            img,
            target_size=(512, 512),
            padding_percent=0.15,
            otsu_offset=OTSU_OFFSET,
        )

        # Step 4 – Wrist Removal
        img, wrist_cut_col, wrist_side = pp.remove_wrist_geometric(
            img,
            otsu_offset=OTSU_OFFSET,
            junction_rise=WRIST_JUNCTION_RISE,
        )

        # Step 5 – Finger Removal (MCP Knuckle Line)
        img, _ = pp.remove_fingers_mcp(
            img,
            wrist_cut_col=wrist_cut_col,
            wrist_side=wrist_side,
            otsu_offset=OTSU_OFFSET,
            safety_margin=MCP_SAFETY_MARGIN,
            defect_depth_min=MCP_DEFECT_DEPTH_MIN,
        )

        # Step 6 – Otsu Segmentation  (get mask only — do NOT apply it yet)
        mask, _ = pp.segment_hand_otsu(img, otsu_offset=OTSU_OFFSET)

        # Step 7 – CLAHE Pre-Enhancement  (on raw metacarpal — no hard edges)
        img = pp.apply_clahe(img,
                             clip_limit=CLAHE_PRE_CLIP,
                             tile_size=CLAHE_TILE_SIZE,
                             label="pre-Sato")

        # Step 8 – Sato Vesselness Filter  (sees smooth image, not hard-cut mask)
        img = pp.apply_sato_filter(
            img,
            scale_min=Sato_SCALE_MIN,
            scale_max=Sato_SCALE_MAX,
            scale_step=Sato_SCALE_STEP,
            black_ridges=True,
        )

        # Step 9 – Apply feathered mask THEN CLAHE post
        # Feathered mask must come AFTER Sato — applying mask before Sato
        # creates hard zero-boundaries that Sato detects as strong ridges,
        # producing the bright rectangular border artifact.
        img = pp.apply_feathered_mask(img, mask, fade_px=12)
        img = pp.apply_clahe(img,
                             clip_limit=CLAHE_POST_CLIP,
                             tile_size=CLAHE_TILE_SIZE,
                             label="post-Sato")

        # Step 10 – Resize to 128 x 128
        img = pp.pad_and_resize(img, mask, target_size=TARGET_SIZE)

        # Step 11 – Z-score Normalization
        img_norm = pp.normalize_image(img)

        # Rescale float back to uint8 for saving as JPEG
        lo, hi = img_norm.min(), img_norm.max()
        img_save = ((img_norm - lo) / (hi - lo + 1e-10) * 255).astype(np.uint8)

        # Ensure destination directory exists and write
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_path), img_save)

        return True, None

    except Exception:
        return False, f"FAILED {src_path}:\n{traceback.format_exc()}"


def collect_jobs(dataset_dir: Path, output_dir: Path) -> tuple[list, int]:
    """
    Return ((src, dst, person_name) triples, total_person_count).
    Groups images by person so we can track per-person progress.
    """
    jobs = []
    person_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    for person_dir in person_dirs:
        for img_file in sorted(person_dir.glob("*.jpg")):
            relative = img_file.relative_to(dataset_dir)
            dst_path = output_dir / relative
            jobs.append((img_file, dst_path, person_dir.name))
    return jobs, len(person_dirs)


def run(dataset_dir: Path, output_dir: Path, workers: int) -> None:
    jobs, total_persons = collect_jobs(dataset_dir, output_dir)
    if not jobs:
        print("No .jpg images found – check the dataset path.")
        sys.exit(1)

    import time

    failures         = []
    persons_done     = 0
    images_done      = 0
    total_images     = len(jobs)
    person_img_ok    = {}
    person_img_total = {}
    last_person      = "—"
    start_time       = time.time()

    for _, _, pname in jobs:
        person_img_total[pname] = person_img_total.get(pname, 0) + 1

    NUM_LINES = 8  # number of lines in the display block

    def _clear_lines(n):
        """Move cursor up n lines and clear each one."""
        for _ in range(n):
            print("\033[A\033[2K", end="")

    def _bar(done, total, width=30):
        filled = int(width * done / total) if total else 0
        return f"[{'█' * filled}{'░' * (width - filled)}]"

    def _fmt_time(seconds):
        seconds = int(seconds)
        h, r = divmod(seconds, 3600)
        m, s = divmod(r, 60)
        return f"{h}h {m:02}m {s:02}s" if h else f"{m}m {s:02}s"

    def _print_progress(first=False):
        elapsed   = time.time() - start_time
        img_rate  = images_done / elapsed if elapsed > 0 else 0
        imgs_left = total_images - images_done
        eta_secs  = imgs_left / img_rate if img_rate > 0 else 0

        person_pct = (persons_done / total_persons * 100) if total_persons else 0
        image_pct  = (images_done  / total_images  * 100) if total_images  else 0

        if not first:
            _clear_lines(NUM_LINES)

        print(f"  ┌─────────────────────────────────────────┐")
        print(f"  │  Persons   {_bar(persons_done, total_persons)}  {persons_done:>4}/{total_persons}  ({person_pct:5.1f}%)  │")
        print(f"  │  Images    {_bar(images_done,  total_images )}  {images_done:>4}/{total_images}  ({image_pct:5.1f}%)  │")
        print(f"  │                                           │")
        print(f"  │  Last finished : {last_person:<24} │")
        print(f"  │  Speed         : {img_rate:>6.2f} img/s              │")
        print(f"  │  Elapsed       : {_fmt_time(elapsed):<26} │")
        print(f"  └─────────────────────────────────────────┘")

    _print_progress(first=True)

    def _handle(pname, success, msg):
        nonlocal persons_done, images_done, last_person
        images_done += 1
        if success:
            person_img_ok[pname] = person_img_ok.get(pname, 0) + 1
            if person_img_ok[pname] == person_img_total[pname]:
                persons_done += 1
                last_person = pname
        else:
            failures.append(msg)
        _print_progress()

    if workers == 1:
        for src, dst, pname in jobs:
            success, msg = preprocess_single(src, dst)
            _handle(pname, success, msg)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(preprocess_single, src, dst): pname
                          for src, dst, pname in jobs}
            for future in as_completed(future_map):
                pname = future_map[future]
                success, msg = future.result()
                _handle(pname, success, msg)

    elapsed = time.time() - start_time
    print(f"\n  ✓ Done — {persons_done} persons, {images_done} images in {_fmt_time(elapsed)}")
    if failures:
        print(f"\n  Failed ({len(failures)}):")
        for f in failures:
            print(f"    • {f.splitlines()[0]}")


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-preprocess the dorsal hand vein dataset."
    )
    parser.add_argument(
        "--dataset", default="dataset",
        help="Path to the source dataset folder (default: dataset)"
    )
    parser.add_argument(
        "--output", default="processed_dataset",
        help="Path to the output folder (default: processed_dataset)"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1)."
    )
    args = parser.parse_args()

    run(
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.output),
        workers=args.workers,
    )