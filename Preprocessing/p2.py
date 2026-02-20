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
  8. Frangi Vesselness Filter  (pure NumPy/SciPy, no DLL issues)
  9. CLAHE post-enhancement
 10. Resize to 224x224
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

IMAGE_PATH  = "C:\\Users\\adith\\OneDrive\\Documents\\py\\Pulse_id\\dataset\\person309\\vein_1.jpg"
OUTPUT_PATH = "C:\\Users\\adith\\OneDrive\\Documents\\py\\Pulse_id\\image\\vein_processed.jpg"

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

# -- Frangi filter -------------------------------------------------------------
FRANGI_SCALE_MIN  = 1    # minimum vein thickness to detect (pixels)
FRANGI_SCALE_MAX  = 10   # maximum vein thickness to detect (pixels)
FRANGI_SCALE_STEP = 1    # step between scales (1 = thorough, 2 = faster)
FRANGI_ALPHA      = 0.5  # plate-like vs line-like sensitivity
FRANGI_BETA       = 0.5  # blob suppression

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

# -- CLAHE ---------------------------------------------------------------------
CLAHE_PRE_CLIP  = 3.0  # clip limit for pre-Frangi CLAHE (boosts NIR contrast)
CLAHE_POST_CLIP = 2.0  # clip limit for post-Frangi CLAHE
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


def remove_wrist_geometric(img, otsu_offset=0, junction_rise=0.15):
    """
    Remove the wrist / forearm using a column-height profile analysis.

    Works purely with OpenCV and NumPy -- no external model needed.
    Automatically detects which side the wrist enters from (left or right).

    Args:
        img           : Cropped grayscale image (post smart-crop).
        otsu_offset   : Same offset used for segmentation.
        junction_rise : Where to cut relative to the wrist-to-palm width rise.
                        0.15 = slightly above the wrist crease (safe default).
    Returns:
        img_no_wrist  : Image with wrist region zeroed out.
        cut_col       : Column where the cut was made.
        orientation   : 'left' or 'right' -- which side the wrist entered from.
    """
    t0 = time.time()
    h, w = img.shape

    otsu_val, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adj = int(np.clip(otsu_val + otsu_offset, 0, 255))
    _, binary = cv2.threshold(img, adj, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("  WARNING: No contour found -- skipping wrist removal")
        return img.copy(), None, None

    hand_mask = np.zeros_like(binary)
    cv2.drawContours(hand_mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)

    col_heights = np.array([np.sum(hand_mask[:, c] > 0) for c in range(w)],
                           dtype=np.float32)
    smoothed    = uniform_filter1d(col_heights, size=max(10, w // 50))

    nz         = np.where(col_heights > 0)[0]
    hand_start = int(nz[0])
    hand_end   = int(nz[-1])
    span       = hand_end - hand_start

    palm_peak = float(np.max(smoothed[hand_start:hand_end]))

    # Detect wrist side:
    # The wrist is a SOLID rectangular entry — its first columns are consistently
    # wider than the fingertip side (which tapers off as scattered narrow protrusions).
    # Compare the MINIMUM column height in the first 15% vs last 15% of the hand span.
    # The wrist side has a higher minimum (solid cylinder), fingertip side has a lower
    # minimum (gaps between fingers).
    check_len  = max(5, int(span * 0.15))
    left_min   = float(np.min(smoothed[hand_start:hand_start + check_len]))
    right_min  = float(np.min(smoothed[hand_end - check_len:hand_end]))
    # Also check the mean of those windows
    left_mean  = float(np.mean(smoothed[hand_start:hand_start + check_len]))
    right_mean = float(np.mean(smoothed[hand_end - check_len:hand_end]))
    # Wrist side has both higher min and higher mean (solid entry)
    left_score  = left_min  + left_mean
    right_score = right_min + right_mean
    wrist_left  = left_score >= right_score
    side = 'left' if wrist_left else 'right'
    print(f"  Wrist enters from the {side.upper()} side  "
          f"(left_score={left_score:.0f}  right_score={right_score:.0f}  "
          f"palm_peak={palm_peak:.0f})")

    if wrist_left:
        # Scan left-to-right; plateau is at the beginning
        half_end   = hand_start + span // 2
        plateau_h  = float(np.median(smoothed[hand_start:hand_start + max(1, span // 6)]))
        cut_thresh = plateau_h + (palm_peak - plateau_h) * junction_rise
        junction   = None
        for c in range(hand_start, half_end):
            if smoothed[c] > cut_thresh:
                junction = c
                break
        if junction is None:
            junction = hand_start + span // 4
        cut_col = max(hand_start, junction - int(w * 0.015))
        img_no_wrist = img.copy()
        img_no_wrist[:, :cut_col] = 0
        print(f"  Junction col={junction}  ->  cut at col={cut_col} (zeroing left of cut)")
    else:
        # Scan right-to-left; plateau is at the end
        half_start = hand_end - span // 2
        plateau_h  = float(np.median(smoothed[hand_end - max(1, span // 6):hand_end]))
        cut_thresh = plateau_h + (palm_peak - plateau_h) * junction_rise
        junction   = None
        for c in range(hand_end, half_start, -1):
            if smoothed[c] > cut_thresh:
                junction = c
                break
        if junction is None:
            junction = hand_end - span // 4
        cut_col = min(hand_end, junction + int(w * 0.015))
        img_no_wrist = img.copy()
        img_no_wrist[:, cut_col:] = 0
        print(f"  Junction col={junction}  ->  cut at col={cut_col} (zeroing right of cut)")

    print(f"  Wrist removal done in {time.time()-t0:.3f}s")
    return img_no_wrist, cut_col, side



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
    # FALLBACK: column-segment counting  (no model needed)
    # ==================================================================
    print("  Using fallback: column-segment counting")

    otsu_val, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adj = int(np.clip(otsu_val + otsu_offset, 0, 255))
    _, binary = cv2.threshold(img, adj, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("  WARNING: No contour found — skipping finger removal")
        return img.copy(), None

    hand_mask = np.zeros_like(binary)
    cv2.drawContours(hand_mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)

    def count_segments(arr):
        in_seg, count = False, 0
        for px in arr:
            if px > 0 and not in_seg: count += 1; in_seg = True
            elif px == 0:             in_seg = False
        return count

    horizontal = wrist_side in ('left', 'right')
    profile    = ([np.sum(hand_mask[:, c] > 0) for c in range(w)] if horizontal
                  else [np.sum(hand_mask[r, :] > 0) for r in range(h)])
    nz         = np.where(np.array(profile) > 0)[0]

    if len(nz) < 10:
        print("  WARNING: Hand mask too sparse — skipping finger removal")
        return img.copy(), None

    entry_idx = int(nz[0])
    exit_idx  = int(nz[-1])
    scan_fwd  = wrist_side in ('left', 'top')
    scan_rng  = (range(entry_idx + 5, exit_idx - 5) if scan_fwd
                 else range(exit_idx - 5, entry_idx + 5, -1))
    step      = 1 if scan_fwd else -1

    split_idx = None
    for idx in scan_rng:
        get = lambda i: (count_segments(hand_mask[:, i]) if horizontal
                         else count_segments(hand_mask[i, :]))
        if get(idx) > 1 and get(idx + step) > 1 and get(idx + 2*step) > 1:
            split_idx = idx
            break

    if split_idx is None:
        print("  WARNING: No finger split found — skipping finger removal")
        return img.copy(), None

    wrist_bound = wrist_cut_col if wrist_cut_col is not None else entry_idx
    if wrist_side == 'left':
        mcp_cut = max(wrist_bound, split_idx - safety_px)
    elif wrist_side == 'right':
        mcp_cut = min(wrist_bound, split_idx + safety_px)
    elif wrist_side == 'top':
        mcp_cut = max(wrist_bound, split_idx - safety_px)
    else:
        mcp_cut = min(wrist_bound, split_idx + safety_px)

    img_out, mcp_cut = apply_cut(mcp_cut)
    print(f"  Fallback: split at idx={split_idx}  safety={safety_px}px  cut={mcp_cut}")
    print(f"  Finger removal done in {time.time()-t0:.3f}s  [fallback]")
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


def _frangi_pure(img_float, sigmas, alpha=0.5, beta=0.5, black_ridges=True):
    """
    Pure NumPy/SciPy implementation of the Frangi vesselness filter.
    Identical mathematics to skimage.filters.frangi — no compiled C extensions.

    For each scale sigma:
      1. Smooth with Gaussian
      2. Compute the 2-D Hessian via second derivatives
      3. Get eigenvalues lambda1, lambda2  (|lambda1| <= |lambda2|)
      4. Compute vesselness score V = exp(-Rb²/2β²) * (1 - exp(-S²/2c²))
         where Rb = lambda1/lambda2  (blob vs line measure)
               S  = sqrt(lambda1²+lambda2²)  (background measure)
               c  = half the max S (auto gamma)
    Responses across scales are max-pooled.
    """
    from scipy.ndimage import gaussian_filter

    vesselness = np.zeros_like(img_float)

    for sigma in sigmas:
        # Gaussian-smoothed second derivatives (Hessian elements)
        # Uses sigma² scaling so responses are scale-invariant
        scale = sigma ** 2
        Dxx = gaussian_filter(img_float, sigma, order=(2, 0)) * scale
        Dyy = gaussian_filter(img_float, sigma, order=(0, 2)) * scale
        Dxy = gaussian_filter(img_float, sigma, order=(1, 1)) * scale

        if black_ridges:
            Dxx, Dyy, Dxy = -Dxx, -Dyy, -Dxy

        # Eigenvalues of the 2×2 symmetric Hessian at each pixel
        # lambda = (Dxx+Dyy)/2 ± sqrt(((Dxx-Dyy)/2)² + Dxy²)
        tmp   = np.sqrt(((Dxx - Dyy) / 2) ** 2 + Dxy ** 2)
        lam1  = (Dxx + Dyy) / 2 - tmp   # smaller eigenvalue
        lam2  = (Dxx + Dyy) / 2 + tmp   # larger eigenvalue

        # Only pixels where lambda2 > 0 are vessel-like (bright ridge)
        vessel_mask = lam2 > 0

        # Rb: anisotropy (line vs blob), S: magnitude
        Rb = np.where(vessel_mask, np.abs(lam1) / (np.abs(lam2) + 1e-10), 0.0)
        S  = np.sqrt(lam1 ** 2 + lam2 ** 2)

        # Auto gamma: half the max structure magnitude at this scale
        c  = 0.5 * S.max() + 1e-10

        V  = np.exp(-(Rb ** 2) / (2 * beta ** 2)) * \
             (1.0 - np.exp(-(S ** 2) / (2 * c ** 2)))
        V[~vessel_mask] = 0.0

        vesselness = np.maximum(vesselness, V)

    return vesselness


def apply_frangi_filter(img, scale_min=1, scale_max=10, scale_step=1,
                        alpha=0.5, beta=0.5, black_ridges=True):
    """
    Apply Frangi vesselness filter using a pure NumPy/SciPy implementation.

    Avoids all scikit-image compiled extensions — works on any platform
    including Windows environments where DLL loading may fail.

    For NIR dorsal-hand images the veins appear as dark ridges on a brighter
    skin background after CLAHE enhancement, so black_ridges=True is correct.
    """
    t0     = time.time()
    img_f  = img.astype(np.float32) / 255.0
    sigmas = np.arange(scale_min, scale_max + scale_step, scale_step)

    print(f"  Running Frangi (pure NumPy/SciPy, no DLL required)")
    print(f"  Scales {scale_min}-{scale_max} step {scale_step}  "
          f"alpha={alpha}  beta={beta}  black_ridges={black_ridges}")

    ves    = _frangi_pure(img_f, sigmas, alpha=alpha, beta=beta,
                         black_ridges=black_ridges)
    ves    = (ves - ves.min()) / (ves.max() - ves.min() + 1e-10)
    result = (ves * 255).astype(np.uint8)

    print(f"  Frangi done in {time.time()-t0:.3f}s")
    return result


def resize_image(img, target_size=224):
    """Resize to CNN input size."""
    resized = cv2.resize(img, (target_size, target_size),
                         interpolation=cv2.INTER_CUBIC)
    print(f"  Resized to {target_size}x{target_size}")
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
    Display every pipeline stage in a grid with a dark background.

    Args:
        stages: list of (image_array, title_string) tuples.
    """
    n     = len(stages)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 4))
    fig.patch.set_facecolor('#0d1117')
    axes_flat = list(axes.flat) if hasattr(axes, 'flat') else [axes]

    for ax, (img, title) in zip(axes_flat, stages):
        # For float normalized images, rescale display to [0, 1]
        display = img
        if img.dtype in (np.float32, np.float64):
            lo, hi = img.min(), img.max()
            display = (img - lo) / (hi - lo + 1e-10)
        ax.imshow(display, cmap='gray', interpolation='nearest')
        ax.set_title(title, color='white', fontsize=10,
                     fontweight='bold', pad=5)
        ax.axis('off')

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.suptitle('Dorsal Hand Vein -- Preprocessing Pipeline',
                 color='white', fontsize=14, fontweight='bold', y=1.01)
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
    print(f"  FRANGI scales        = {FRANGI_SCALE_MIN}-{FRANGI_SCALE_MAX} "
          f"step {FRANGI_SCALE_STEP}")
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

    print("\n[Step 6] Otsu Segmentation")
    mask, img_segmented = segment_hand_otsu(img_metacarpal, otsu_offset=OTSU_OFFSET)

    print("\n[Step 7] CLAHE Pre-Enhancement")
    img_clahe_pre = apply_clahe(img_segmented,
                                clip_limit=CLAHE_PRE_CLIP,
                                tile_size=CLAHE_TILE_SIZE,
                                label="pre-Frangi")

    print("\n[Step 8] Frangi Filter")
    img_frangi = apply_frangi_filter(
        img_clahe_pre,
        scale_min=FRANGI_SCALE_MIN,
        scale_max=FRANGI_SCALE_MAX,
        scale_step=FRANGI_SCALE_STEP,
        alpha=FRANGI_ALPHA,
        beta=FRANGI_BETA,
        black_ridges=True,
    )

    print("\n[Step 9] CLAHE Post-Enhancement")
    img_clahe_post = apply_clahe(img_frangi,
                                 clip_limit=CLAHE_POST_CLIP,
                                 tile_size=CLAHE_TILE_SIZE,
                                 label="post-Frangi")

    print("\n[Step 10] Resize")
    img_resized = resize_image(img_clahe_post, target_size=224)

    print("\n[Step 11] Normalize")
    img_normalized = normalize_image(img_resized)

    # Save (rescale normalized image back to uint8 for disk)
    img_save = (
        (img_normalized - img_normalized.min()) /
        (img_normalized.max() - img_normalized.min()) * 255
    ).astype(np.uint8)
    cv2.imwrite(OUTPUT_PATH, img_save)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Final shape: {img_normalized.shape}  dtype: {img_normalized.dtype}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    # Show all pipeline stages
    stages = [
        (img_original,   "1. Original"),
        (img_blurred,    "2. Gaussian Blur"),
        (img_cropped,    "3. Smart Crop (512x512)"),
        (img_no_wrist,   "4. Wrist Removed"),
        (img_metacarpal, "5. Fingers Removed (MCP)"),
        (mask,           "6. Otsu Mask"),
        (img_segmented,  "7. Segmented"),
        (img_clahe_pre,  "8. CLAHE Pre-Frangi"),
        (img_frangi,     "9. Frangi Filter"),
        (img_clahe_post, "10. CLAHE Post-Frangi"),
        (img_resized,    "11. Resized 224x224"),
        (img_save,       "12. Normalized (display)"),
    ]
    visualize_pipeline(stages)