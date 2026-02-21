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
# ===========================================================================


def preprocess_single(src_path: Path, dst_path: Path) -> tuple[bool, str]:
    """
    Run the full 11-step pipeline on one image and save the result.

    Returns (success: bool, message: str).
    """
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

        # Step 6 – Otsu Segmentation
        _, img = pp.segment_hand_otsu(img, otsu_offset=OTSU_OFFSET)

        # Step 7 – CLAHE Pre-Enhancement
        img = pp.apply_clahe(img,
                             clip_limit=CLAHE_PRE_CLIP,
                             tile_size=CLAHE_TILE_SIZE,
                             label="pre-Sato")

        # Step 8 – Sato Vesselness Filter
        img = pp.apply_sato_filter(
            img,
            scale_min=Sato_SCALE_MIN,
            scale_max=Sato_SCALE_MAX,
            scale_step=Sato_SCALE_STEP,
            black_ridges=True,
        )

        # Step 9 – CLAHE Post-Enhancement
        img = pp.apply_clahe(img,
                             clip_limit=CLAHE_POST_CLIP,
                             tile_size=CLAHE_TILE_SIZE,
                             label="post-Sato")

        # Step 10 – Resize to 224 × 224
        img = pp.resize_image(img, target_size=224)

        # Step 11 – Z-score Normalization
        img_norm = pp.normalize_image(img)

        # Rescale float back to uint8 for saving as JPEG
        lo, hi = img_norm.min(), img_norm.max()
        img_save = ((img_norm - lo) / (hi - lo + 1e-10) * 255).astype(np.uint8)

        # Ensure destination directory exists and write
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_path), img_save)

        return True, f"OK  {src_path.name}"

    except Exception:
        return False, f"FAILED {src_path}:\n{traceback.format_exc()}"


def collect_jobs(dataset_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    """Return (src, dst) pairs for every .jpg found under dataset_dir."""
    jobs = []
    for person_dir in sorted(dataset_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        for img_file in sorted(person_dir.glob("*.jpg")):
            relative   = img_file.relative_to(dataset_dir)
            dst_path   = output_dir / relative
            jobs.append((img_file, dst_path))
    return jobs


def run(dataset_dir: Path, output_dir: Path, workers: int, quiet: bool) -> None:
    jobs = collect_jobs(dataset_dir, output_dir)
    total = len(jobs)
    if total == 0:
        print("No .jpg images found – check the dataset path.")
        sys.exit(1)

    print(f"Found {total} images across {len(list(dataset_dir.iterdir()))} persons.")
    print(f"Output → {output_dir.resolve()}")
    print(f"Workers: {workers}\n")

    ok_count = fail_count = 0
    failures = []

    if workers == 1:
        # Single-process – easier to debug
        for i, (src, dst) in enumerate(jobs, 1):
            success, msg = preprocess_single(src, dst)
            if success:
                ok_count += 1
                if not quiet:
                    print(f"[{i:>5}/{total}] {msg}")
            else:
                fail_count += 1
                failures.append(msg)
                print(f"[{i:>5}/{total}] {msg}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(preprocess_single, src, dst): (i + 1, src)
                          for i, (src, dst) in enumerate(jobs)}
            for future in as_completed(future_map):
                idx, src = future_map[future]
                success, msg = future.result()
                if success:
                    ok_count += 1
                    if not quiet:
                        print(f"[{idx:>5}/{total}] {msg}")
                else:
                    fail_count += 1
                    failures.append(msg)
                    print(f"[{idx:>5}/{total}] {msg}", file=sys.stderr)

    # Summary
    print("\n" + "=" * 60)
    print(f"  Completed : {ok_count}/{total}")
    print(f"  Failed    : {fail_count}")
    if failures:
        print("\nFailed images:")
        for f in failures:
            print("  •", f.splitlines()[0])
    print("=" * 60)


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
        help="Number of parallel worker processes (default: 1). "
             "Use >1 for faster bulk processing on multi-core machines."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-image success messages."
    )
    args = parser.parse_args()

    run(
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.output),
        workers=args.workers,
        quiet=args.quiet,
    )
