#!/usr/bin/env python3
"""
Dorsal Hand Vein Preprocessing Pipeline

REQUIREMENTS:
    pip install opencv-python numpy matplotlib scikit-image scipy

Pipeline:
  1. Load (grayscale)
  2. Gaussian Blur
  3. Smart Crop  (Otsu-based hand detection + centering)
  4. Wrist Removal  (column-profile geometry, no external model needed)
  5. Finger Removal  (MCP knuckle line via convexity defects)
  6. Otsu Segmentation
  7. CLAHE pre-enhancement  (boosts low-contrast NIR vein structure)
  8. Sato Vesselness Filter  (pure NumPy/SciPy, no DLL issues)
  9. CLAHE post-enhancement
 10. Tight crop ROI + Resize to TARGET_SIZE
 11. Z-score Normalization
"""        

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.ndimage import uniform_filter1d


# ==============================================================================
#  CONFIGURATION  -- edit these values to tune the pipeline
# ==============================================================================

IMAGE_PATH  = "C:\\Users\\adith\\OneDrive\\Documents\\py\\Pulse_id\\dataset\\person15\\vein_1.jpg"
OUTPUT_PATH = "C:\\Users\\adith\\OneDrive\\Documents\\py\\Pulse_id\\image\\vein_processed.png"

# -- Otsu threshold offset -----------------------------------------------------
#   0   = pure Otsu (automatic)
#  +N   = stricter  -> removes more dim background pixels
#  -N   = looser    -> keeps more hand-edge pixels (good for dark NIR images)
OTSU_OFFSET = -20

# -- Wrist removal -------------------------------------------------------------
#   Fraction of the column-height rise from wrist plateau to palm peak at which
#   to place the cut line.
#   0.15 = cut where the hand has widened 15% above the wrist plateau (safe)
#   Lower = tighter cut (closer to wrist crease)
#   Higher = more conservative cut (removes more wrist)
WRIST_JUNCTION_RISE = 0.15

# -- Sato filter -------------------------------------------------------------
Sato_SCALE_MIN  = 3    # minimum vein thickness to detect (pixels)
Sato_SCALE_MAX  = 3    # maximum vein thickness to detect (pixels)
Sato_SCALE_STEP = 1    # step between scales (1 = thorough, 2 = faster)


# -- Finger removal ------------------------------------------------------------
#   Uses convexity defects to locate the MCP knuckle line (where fingers begin
#   to separate from the palm body) and removes everything beyond it.
#   Only the stable metacarpal vein region is kept for biometric matching.
#
#   MCP_SAFETY_MARGIN    : fraction of image width shifted toward the wrist from
#                          the raw knuckle line.  0.05 keeps full knuckle tissue.
#                          Increase to keep more,  decrease for a tighter cut.
#   MCP_DEFECT_DEPTH_MIN : minimum convexity defect depth (px) treated as a
#                          real finger valley.  Raise if false valleys appear.
MCP_SAFETY_MARGIN    = 0.03
MCP_DEFECT_DEPTH_MIN = 20    # pixels

# -- Output size ---------------------------------------------------------------
#   128 is recommended for Pi deployment: small model, fast inference.
#   Tight-crop is applied before resize so no black padding is wasted.
#   Increase to 160 or 224 if accuracy matters more than speed.
TARGET_SIZE = 128

# -- CLAHE ---------------------------------------------------------------------
CLAHE_PRE_CLIP  = 1.5  # clip limit for pre-Sato CLAHE (boosts NIR contrast)
CLAHE_POST_CLIP = 0.5  # clip limit for post-Sato CLAHE
CLAHE_TILE_SIZE = 8    # tile grid size for both passes

# ==============================================================================


def load_image(image_path):
    """Load image in grayscale."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    print(f"  Loaded: {img.shape[1]}x{img.shape[0]}  max={img.max()}  mean={img.mean():.1f}")
    return img


def apply_gaussian_blur(img, kernel_size=11):
    """Apply Gaussian blur to reduce sensor noise."""
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    print(f"  Gaussian blur  kernel={kernel_size}x{kernel_size}")
    return blurred


def smart_crop_hand_region(img, target_size=(512, 512),
                           padding_percent=0.15, otsu_offset=0):
    """
    Detect the hand bounding box using Otsu thresholding, centre it, and
    resize to target_size.

    Args:
        img             : Blurred grayscale image.
        target_size     : Output (width, height).
        padding_percent : Extra padding around the hand bbox (0.15 = 15%).
        otsu_offset     : Integer offset added to the Otsu threshold.
    """
    t0 = time.time()
    h, w = img.shape

    otsu_val, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adj = int(np.clip(otsu_val + otsu_offset, 0, 255))
    if otsu_offset != 0:
        print(f"  Otsu={otsu_val:.0f}  offset={otsu_offset:+d}  -> using {adj}")
    else:
        print(f"  Otsu threshold={otsu_val:.0f}")
    _, binary = cv2.threshold(img, adj, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("  WARNING: No hand detected -- using centre crop fallback")
        s    = min(w, h)
        crop = img[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
        return cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC), {}

    bx, by, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cx, cy = bx + bw // 2, by + bh // 2
    print(f"  Hand bbox: ({bx},{by})  {bw}x{bh}  centre=({cx},{cy})")

    size = int(max(bw, bh) * (1 + 2 * padding_percent))
    x1   = max(0, cx - size // 2)
    y1   = max(0, cy - size // 2)
    x2   = min(w, x1 + size)
    y2   = min(h, y1 + size)
    x1   = max(0, x2 - size)
    y1   = max(0, y2 - size)

    cropped = img[y1:y2, x1:x2]

    hx_off = cx - x1 - cropped.shape[1] // 2
    hy_off = cy - y1 - cropped.shape[0] // 2
    if abs(hx_off) > 1 or abs(hy_off) > 1:
        M      = np.float32([[1, 0, -hx_off], [0, 1, -hy_off]])
        cropped = cv2.warpAffine(cropped, M, (cropped.shape[1], cropped.shape[0]),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    cropped = cv2.resize(cropped, target_size, interpolation=cv2.INTER_CUBIC)
    print(f"  Smart crop done in {time.time()-t0:.3f}s  ->  {target_size[0]}x{target_size[1]}")
    return cropped, {'bbox': (bx, by, bw, bh), 'otsu': adj}


def remove_wrist_geometric(img, otsu_offset=0, junction_rise=0.2):
    """
    Wrist removal using distance-transform-based thickness profiling.
    Assumes FIXED orientation:
        - Wrist on LEFT
        - Fingers on RIGHT

    Robust to hair, boundary noise, and weak silhouettes.

    Args:
        img           : Cropped grayscale image (uint8).
        otsu_offset   : Offset added to Otsu threshold.
        junction_rise : Fraction of wrist-to-palm thickness rise at which to cut.
                        0.2 is a safe default.

    Returns:
        img_no_wrist  : Image with wrist region zeroed out.
        cut_col       : Column where wrist was removed.
        wrist_side    : Always 'left'
    """
    import time
    import cv2
    import numpy as np
    from scipy.ndimage import uniform_filter1d

    t0 = time.time()
    h, w = img.shape

    # ------------------------------------------------------------
    # 1. Binary hand mask (hair noise is OK)
    # ------------------------------------------------------------
    otsu_val, _ = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    adj = int(np.clip(otsu_val + otsu_offset, 0, 255))
    _, binary = cv2.threshold(img, adj, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        print("  WARNING: No contour found — skipping wrist removal")
        return img.copy(), None, 'left'

    hand_mask = np.zeros_like(binary)
    cv2.drawContours(
        hand_mask,
        [max(contours, key=cv2.contourArea)],
        -1, 255, -1
    )

    # ------------------------------------------------------------
    # 2. Distance transform → interior thickness
    # ------------------------------------------------------------
    dist = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)

    # Thickness profile = max inscribed radius per column
    thickness = np.array(
        [dist[:, c].max() for c in range(w)], dtype=np.float32
    )

    # Smooth profile (important)
    thickness = uniform_filter1d(thickness, size=max(10, w // 40))

    nz = np.where(thickness > 0)[0]
    if len(nz) < 10:
        print("  WARNING: Hand region too small — skipping wrist removal")
        return img.copy(), None, 'left'

    hand_start = nz[0]
    hand_end   = nz[-1]
    span       = hand_end - hand_start

    # ------------------------------------------------------------
    # 3. Wrist → palm junction via thickness rise
    # ------------------------------------------------------------
    check_len = max(5, int(span * 0.15))

    wrist_thick = float(
        np.median(thickness[hand_start : hand_start + check_len])
    )
    palm_peak = float(thickness.max())

    cut_thresh = wrist_thick + (palm_peak - wrist_thick) * junction_rise

    junction = None
    for c in range(hand_start + check_len, hand_start + span // 2):
        if thickness[c] > cut_thresh:
            junction = c
            break

    if junction is None:
        junction = hand_start + span // 4

    # Small safety offset toward wrist
    cut_col = max(hand_start, junction - int(w * 0.015))

    img_no_wrist = img.copy()
    img_no_wrist[:, :cut_col] = 0

    print(
        f"  Wrist removed at column {cut_col}  "
        f"(wrist_thick={wrist_thick:.1f}, palm_peak={palm_peak:.1f})"
    )
    print(
        f"  Wrist removal done in {time.time() - t0:.3f}s  "
        f"[distance-transform]"
    )

    return img_no_wrist, cut_col, 'left'



def remove_fingers_mcp(img, wrist_cut_col, wrist_side,
                       otsu_offset=0,
                       safety_margin=0.03,
                       defect_depth_min=20):
    """
    Remove fingers by locating the MCP knuckle line.

    PRIMARY METHOD  — MediaPipe HandLandmarker
    ==========================================
    Uses MediaPipe's 21-point hand landmark model to locate the 4 MCP knuckle
    joints (landmarks 5, 9, 13, 17 = index/middle/ring/pinky bases).
    These are the exact anatomical knuckle points — no thresholding involved.

    A cut line is fitted through the 4 MCP points using linear regression so
    it follows the natural slight diagonal of the knuckle row.  The cut is
    offset toward the wrist by safety_margin to keep all knuckle tissue.

    FALLBACK METHOD — column-segment counting (no model required)
    =============================================================
    Used automatically if:
      • mediapipe is not installed, OR
      • the model fails to detect a hand (bad lighting, low contrast, etc.)

    Scans the hand mask column-by-column from the wrist.  The MCP line is the
    first column where the silhouette stably splits into 2+ disconnected
    segments (fingers beginning to separate).

    Args:
        img              : Wrist-removed grayscale image (uint8).
        wrist_cut_col    : Column/row of the wrist cut (safety lower bound).
        wrist_side       : 'left' | 'right' | 'top' | 'bottom'
        otsu_offset      : Threshold offset used only by the fallback method.
        safety_margin    : Fraction of image dimension to shift the cut back
                           toward the wrist from the detected knuckle line.
                           0.03 = 3 % (default).  Increase to keep more knuckle.
        defect_depth_min : Unused — kept for API compatibility.

    Returns:
        img_metacarpal : Image with finger region zeroed out.
        mcp_cut        : Pixel coordinate (column or row) of the cut.
    """
    t0   = time.time()
    h, w = img.shape

    # ------------------------------------------------------------------
    # Helper shared by both methods
    # ------------------------------------------------------------------
    def apply_cut(cut_coord):
        """Zero out the finger side and return (img_out, cut_coord)."""
        out = img.copy()
        if wrist_side == 'left':
            out[:, cut_coord:] = 0
        elif wrist_side == 'right':
            out[:, :cut_coord] = 0
        elif wrist_side == 'top':
            out[cut_coord:, :] = 0
        else:  # bottom
            out[:cut_coord, :] = 0
        return out, cut_coord

    safety_px = int((w if wrist_side in ('left', 'right') else h) * safety_margin)

    # ==================================================================
    # PRIMARY: MediaPipe HandLandmarker
    # ==================================================================
    _mp_ok = False
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode
        _mp_ok = True
    except Exception:
        print("\n Import Failed")
        print(Exception)

    # Find the .task model file — look next to this script and in cwd
    _model_path = None
    if _mp_ok:
        import os
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'hand_landmarker.task'),
            os.path.join(os.getcwd(), 'hand_landmarker.task'),
            'hand_landmarker.task',
        ]
        for c in candidates:
            if os.path.isfile(c):
                _model_path = c
                break
        if _model_path is None:
            print("  MediaPipe: hand_landmarker.task not found next to script or in cwd")
            print("  Download: https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            _mp_ok = False

    if _mp_ok:
        try:
            # Convert grayscale → RGB (MediaPipe expects 3-channel)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Normalise low-brightness NIR images so detection is more reliable
            # (stretch to full 0-255 range before passing to the model)
            lo, hi = img_rgb.min(), img_rgb.max()
            if hi > lo:
                img_rgb = ((img_rgb.astype(np.float32) - lo) /
                           (hi - lo) * 255).astype(np.uint8)

            options = HandLandmarkerOptions(
                base_options   = BaseOptions(model_asset_path=_model_path),
                running_mode   = RunningMode.IMAGE,
                num_hands      = 1,
                min_hand_detection_confidence = 0.2,
                min_hand_presence_confidence  = 0.2,
                min_tracking_confidence       = 0.2,
            )
            with mp_vision.HandLandmarker.create_from_options(options) as detector:
                mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result  = detector.detect(mp_img)

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]   # first detected hand

                # MCP joints: 5=index, 9=middle, 13=ring, 17=pinky
                mcp_ids = [5, 9, 13, 17]
                mcp_pts = [(lm[i].x * w, lm[i].y * h) for i in mcp_ids]

                print(f"  MediaPipe: detected hand  "
                      f"MCP points (px): " +
                      "  ".join([f"({p[0]:.0f},{p[1]:.0f})" for p in mcp_pts]))

                # Fit line through MCP points using least-squares
                # For horizontal hands (wrist left/right) use x as the cut coord
                if wrist_side in ('left', 'right'):
                    xs = [p[0] for p in mcp_pts]
                    mcp_raw = int(np.mean(xs))      # mean x of knuckle bases
                else:
                    ys = [p[1] for p in mcp_pts]
                    mcp_raw = int(np.mean(ys))      # mean y of knuckle bases

                # Shift toward wrist by safety margin
                wrist_bound = wrist_cut_col if wrist_cut_col is not None else 0
                if wrist_side == 'left':
                    mcp_cut = max(wrist_bound, mcp_raw - safety_px)
                elif wrist_side == 'right':
                    mcp_cut = min(wrist_bound, mcp_raw + safety_px)
                elif wrist_side == 'top':
                    mcp_cut = max(wrist_bound, mcp_raw - safety_px)
                else:
                    mcp_cut = min(wrist_bound, mcp_raw + safety_px)

                img_out, mcp_cut = apply_cut(mcp_cut)
                print(f"  MediaPipe: MCP raw={mcp_raw}  "
                      f"safety={safety_px}px  cut={mcp_cut}")
                print(f"  Finger removal done in {time.time()-t0:.3f}s  [MediaPipe]")
                return img_out, mcp_cut

            else:
                print("  MediaPipe: no hand detected — using fallback method")

        except Exception as e:
            print(f"  MediaPipe error ({e}) — using fallback method")

    # ==================================================================
    # FALLBACK: Convex Hull + Convexity Defects
    # ==================================================================
    print("  Using fallback: convex hull + convexity defects")

    otsu_val, _ = cv2.threshold(img, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adj = int(np.clip(otsu_val + otsu_offset, 0, 255))
    _, binary = cv2.threshold(img, adj, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("  WARNING: No contour found — skipping finger removal")
        return img.copy(), None

    cnt = max(contours, key=cv2.contourArea)

    # Convex hull (indices required for defects)
    hull = cv2.convexHull(cnt, returnPoints=False)

    if hull is None or len(hull) < 3:
        print("  WARNING: Convex hull failed — skipping finger removal")
        return img.copy(), None

    defects = cv2.convexityDefects(cnt, hull)

    if defects is None:
        print("  WARNING: No convexity defects found — skipping finger removal")
        return img.copy(), None

    # Collect deep defects (finger valleys)
    valley_pts = []
    for d in defects:
        s, e, f, depth = d[0]
        depth_px = depth / 256.0
        if depth_px > defect_depth_min:
            far = cnt[f][0]   # deepest valley point
            valley_pts.append(far)

    if len(valley_pts) < 2:
        print("  WARNING: Insufficient finger valleys — skipping finger removal")
        return img.copy(), None

    valley_pts = np.array(valley_pts)

    # Estimate MCP cut line
    if wrist_side in ('left', 'right'):
        raw_cut = int(np.mean(valley_pts[:, 0]))   # x-coordinate
        safety_px = int(w * safety_margin)
    else:
        raw_cut = int(np.mean(valley_pts[:, 1]))   # y-coordinate
        safety_px = int(h * safety_margin)

    wrist_bound = wrist_cut_col if wrist_cut_col is not None else 0

    if wrist_side == 'left':
        mcp_cut = max(wrist_bound, raw_cut - safety_px)
    elif wrist_side == 'right':
        mcp_cut = min(wrist_bound, raw_cut + safety_px)
    elif wrist_side == 'top':
        mcp_cut = max(wrist_bound, raw_cut - safety_px)
    else:
        mcp_cut = min(wrist_bound, raw_cut + safety_px)

    # Apply cut
    img_out = img.copy()
    if wrist_side == 'left':
        img_out[:, mcp_cut:] = 0
    elif wrist_side == 'right':
        img_out[:, :mcp_cut] = 0
    elif wrist_side == 'top':
        img_out[mcp_cut:, :] = 0
    else:
        img_out[:mcp_cut, :] = 0

    print(f"  Convexity fallback: valleys={len(valley_pts)}  "
        f"raw_cut={raw_cut}  safety={safety_px}px  cut={mcp_cut}")
    print(f"  Finger removal done in {time.time()-t0:.3f}s  [convexity]")

    return img_out, mcp_cut


def segment_hand_otsu(img, otsu_offset=0):
    """Segment hand using Otsu thresholding and contour filtering."""
    t0 = time.time()
    otsu_val, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adj = int(np.clip(otsu_val + otsu_offset, 0, 255))
    if otsu_offset != 0:
        print(f"  Otsu={otsu_val:.0f}  offset={otsu_offset:+d}  -> using {adj}")
    else:
        print(f"  Otsu threshold={otsu_val:.0f}")

    _, binary_mask = cv2.threshold(img, adj, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        clean = np.zeros_like(binary_mask)
        cv2.drawContours(clean, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        binary_mask = clean

    segmented = cv2.bitwise_and(img, img, mask=binary_mask)
    print(f"  Segmentation done in {time.time()-t0:.3f}s")
    return binary_mask, segmented


def apply_clahe(img, clip_limit=2.0, tile_size=8, label=""):
    """Apply CLAHE contrast-limited adaptive histogram equalization."""
    clahe    = cv2.createCLAHE(clipLimit=clip_limit,
                               tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(img)
    tag = f" [{label}]" if label else ""
    print(f"  CLAHE{tag}  clip={clip_limit}  tile={tile_size}x{tile_size}")
    return enhanced


def apply_sato_filter(img,
                      scale_min=1,
                      scale_max=3,
                      scale_step=1,
                      black_ridges=True):
    """
    Apply Sato vesselness filter for palm vein enhancement.

    Better suited than Sato for NIR palm veins:
    - Responds to weak, thin veins
    - Less sensitive to bones / palm curvature
    - More stable on grayscale images

    Args:
        img           : uint8 grayscale image
        scale_min     : minimum vein scale (pixels)
        scale_max     : maximum vein scale (pixels)
        scale_step    : scale step
        black_ridges  : True for dark veins on bright skin

    Returns:
        result : uint8 enhanced image (veins bright)
    """
    import numpy as np
    from skimage.filters import sato

    img_f = img.astype(np.float32) / 255.0

    sigmas = list(range(scale_min, scale_max + 1, scale_step))

    print(f"  Running Sato filter  scales={sigmas}  black_ridges={black_ridges}")

    ves = sato(
        img_f,
        sigmas=sigmas,
        black_ridges=black_ridges
    )

    # Normalize to [0, 255]
    ves = (ves - ves.min()) / (ves.max() - ves.min() + 1e-10)
    result = (ves * 255).astype(np.uint8)

    return result

def apply_feathered_mask(img, mask, fade_px=12):
    """
    Apply hand mask with a soft fade near the boundary.

    Instead of hard-zeroing outside the mask (which creates sharp edges
    that Sato detects as ridges), erode the mask then blur it to create
    a smooth transition from 0 outside to 1 inside.

    Args:
        img     : uint8 grayscale Sato output.
        mask    : Binary hand mask (255 = hand, 0 = background).
        fade_px : Approximate fade width in pixels.

    Returns:
        Masked uint8 image with smooth boundary.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fade_px, fade_px))
    mask_soft = cv2.erode(mask, k, iterations=1)
    mask_soft = cv2.GaussianBlur(mask_soft.astype(np.float32),
                                 (fade_px*2+1, fade_px*2+1), 0) / 255.0
    return (img.astype(np.float32) * mask_soft).astype(np.uint8)


def pad_and_resize(img, mask, target_size=128, padding=6):
    """
    Tight-crop the hand ROI, pad to a square preserving aspect ratio,
    then resize to target_size.

    Avoids two problems:
      1. Black padding waste — without tight crop, hand occupies only ~35%
         of the 512x512 frame so most pixels are wasted background.
      2. Aspect ratio distortion — naive square resize of a non-square ROI
         stretches the vein pattern and changes apparent vein spacing.

    Strategy: crop tight → find longest side → pad shorter side equally
    on both ends with zeros → resize square. The hand stays centred and
    undistorted.

    Args:
        img         : Grayscale image (after Sato + CLAHE).
        mask        : Binary hand mask.
        target_size : Output side length in pixels (default 128).
        padding     : Extra pixels around tight bbox before squaring (default 6).

    Returns:
        Square image of shape (target_size, target_size).
    """
    nz = np.where(mask > 0)
    if len(nz[0]) == 0:
        print(f"  WARNING: empty mask — centre-padding full frame")
        nz = np.where(img > 0)
        if len(nz[0]) == 0:
            return cv2.resize(img, (target_size, target_size))

    h, w  = img.shape
    r_min = max(0, int(nz[0].min()) - padding)
    r_max = min(h, int(nz[0].max()) + padding)
    c_min = max(0, int(nz[1].min()) - padding)
    c_max = min(w, int(nz[1].max()) + padding)

    cropped = img[r_min:r_max, c_min:c_max]
    ch, cw  = cropped.shape

    # Pad shorter dimension to make it square (centre the content)
    side = max(ch, cw)
    pad_top    = (side - ch) // 2
    pad_bottom = side - ch - pad_top
    pad_left   = (side - cw) // 2
    pad_right  = side - cw - pad_left

    squared = cv2.copyMakeBorder(cropped,
                                 pad_top, pad_bottom, pad_left, pad_right,
                                 cv2.BORDER_CONSTANT, value=0)
    resized = cv2.resize(squared, (target_size, target_size),
                         interpolation=cv2.INTER_CUBIC)
    print(f"  ROI: {cw}x{ch}px  padded to {side}x{side}  ->  {target_size}x{target_size}")
    return resized



def normalize_image(img):
    """Z-score normalisation for CNN input."""
    f    = img.astype(np.float32)
    mean = f.mean()
    std  = f.std()
    if std < 1e-6:
        print("  WARNING: low std -- using min-max normalisation")
        out = (f - f.min()) / (f.max() - f.min() + 1e-6)
    else:
        out = (f - mean) / std
    print(f"  Normalised  mean={mean:.2f}  std={std:.2f}  "
          f"range=[{out.min():.2f}, {out.max():.2f}]")
    return out


def visualize_pipeline(stages):
    """
    Display 8 pipeline stages in a 2-row x 4-col grid on a dark background.

    Each panel shows the cumulative result after that group of steps:
      Row 1: Original | Blur+Crop | Wrist+Finger cut | Otsu segmented
      Row 2: CLAHE pre | Sato     | CLAHE post       | Resized+Normalized
    """
    assert len(stages) == 8, f"Expected 8 stages, got {len(stages)}"

    fig, axes = plt.subplots(2, 4, figsize=(4 * 4, 2 * 4.2))
    fig.patch.set_facecolor('#0d1117')

    for ax, (img, title) in zip(axes.flat, stages):
        display = img
        if img.dtype in (np.float32, np.float64):
            lo, hi = img.min(), img.max()
            display = (img - lo) / (hi - lo + 1e-10)
        ax.imshow(display, cmap='gray', interpolation='nearest')
        ax.set_title(title, color='white', fontsize=11,
                     fontweight='bold', pad=6)
        ax.axis('off')

    plt.suptitle('Dorsal Hand Vein — Preprocessing Pipeline',
                 color='white', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    print("\nDisplaying pipeline visualization -- close the window to exit")
    plt.show()


# ==============================================================================
#  MAIN
# ==============================================================================

if __name__ == "__main__":

    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("DORSAL HAND VEIN PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"  OTSU_OFFSET          = {OTSU_OFFSET}")
    print(f"  WRIST_JUNCTION_RISE  = {WRIST_JUNCTION_RISE}")
    print(f"  MCP_SAFETY_MARGIN    = {MCP_SAFETY_MARGIN}")
    print(f"  MCP_DEFECT_DEPTH_MIN = {MCP_DEFECT_DEPTH_MIN}px")
    print(f"  Sato scales        = {Sato_SCALE_MIN}-{Sato_SCALE_MAX} "
          f"step {Sato_SCALE_STEP}")
    print(f"  TARGET_SIZE          = {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"  CLAHE pre/post clip  = {CLAHE_PRE_CLIP} / {CLAHE_POST_CLIP}")
    print("=" * 60)

    print("\n[Step 1 & 2] Load + Gaussian Blur")
    img_original = load_image(IMAGE_PATH)
    img_blurred  = apply_gaussian_blur(img_original, kernel_size=11)

    print("\n[Step 3] Smart Crop")
    img_cropped, _ = smart_crop_hand_region(
        img_blurred,
        target_size=(512, 512),
        padding_percent=0.15,
        otsu_offset=OTSU_OFFSET,
    )

    print("\n[Step 4] Wrist Removal")
    img_no_wrist, wrist_cut_col, wrist_side = remove_wrist_geometric(
        img_cropped,
        otsu_offset=OTSU_OFFSET,
        junction_rise=WRIST_JUNCTION_RISE,
    )

    print("\n[Step 5] Finger Removal (MCP Knuckle Line)")
    img_metacarpal, mcp_cut_col = remove_fingers_mcp(
        img_no_wrist,
        wrist_cut_col=wrist_cut_col,
        wrist_side=wrist_side,
        otsu_offset=OTSU_OFFSET,
        safety_margin=MCP_SAFETY_MARGIN,
        defect_depth_min=MCP_DEFECT_DEPTH_MIN,
    )

    print("\n[Step 6] Otsu Segmentation  (mask only — not applied yet)")
    mask, img_segmented = segment_hand_otsu(img_metacarpal, otsu_offset=OTSU_OFFSET)

    print("\n[Step 7] CLAHE Pre-Enhancement  (on raw metacarpal — no hard edges)")
    img_clahe_pre = apply_clahe(img_metacarpal,
                                clip_limit=CLAHE_PRE_CLIP,
                                tile_size=CLAHE_TILE_SIZE,
                                label="pre-Sato")

    print("\n[Step 8] Sato Filter  (on smooth image — no boundary artifacts)")
    img_Sato = apply_sato_filter(
        img_clahe_pre,
        scale_min=Sato_SCALE_MIN,
        scale_max=Sato_SCALE_MAX,
        scale_step=Sato_SCALE_STEP,
        black_ridges=True,
    )

    print("\n[Step 9] Apply feathered mask + CLAHE post")
    img_Sato      = apply_feathered_mask(img_Sato, mask, fade_px=12)
    img_clahe_post = apply_clahe(img_Sato,
                                 clip_limit=CLAHE_POST_CLIP,
                                 tile_size=CLAHE_TILE_SIZE,
                                 label="post-Sato")

    print("\n[Step 10] Tight Crop + Resize")
    img_resized = pad_and_resize(img_clahe_post, mask, target_size=TARGET_SIZE)

    print("\n[Step 11] Normalize")
    img_normalized = normalize_image(img_resized)

    # Save (rescale normalized image back to uint8 for disk)
    img_save = (
        (img_normalized - img_normalized.min()) /
        (img_normalized.max() - img_normalized.min()) * 255
    ).astype(np.uint8)
    cv2.imwrite(OUTPUT_PATH, img_save, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Final shape: {img_normalized.shape}  dtype: {img_normalized.dtype}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    stages = [
        (img_original,   "1. Original"),
        (img_cropped,    "2. Blur + Crop"),
        (img_metacarpal, "3. Wrist + Finger removed"),
        (img_segmented,  "4. Segmented"),
        (img_clahe_pre,  "5. CLAHE Pre-Sato"),
        (img_Sato,       "6. Sato Filter"),
        (img_clahe_post, "7. CLAHE Post-Sato"),
        (img_save,       f"8. {TARGET_SIZE}x{TARGET_SIZE} Normalized"),
    ]
    visualize_pipeline(stages)