#!/usr/bin/env python3
"""
augmentation.py
===============

    Apply augmentation pipeline to a single vein image.
    Pipeline order:
        1. Spatial transforms  (rotation, translation)
        2. Photometric transforms (brightness, contrast, noise, sharpen/blur)
        3. Spatial transforms ( grid distortion)
        4. Random erasing last  (applied after photometric so erased regions stay black)
"""

import random
import numpy as np
import cv2


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

AUG_CONFIG = {
    # Spatial
    "rotation_deg":       12,     # ± degrees
    "translate_frac":     0.07,   # ± fraction of image size

    # Photometric — reduced since lighting is controlled

    "noise_std":          6,      # gaussian noise std in pixel space (was 8)

    # Random erasing
    "erase_prob":         0.5,    # probability of applying
    "erase_min_frac":     0.02,   # min erased area as fraction of image
    "erase_max_frac":     0.12,   # max erased area as fraction of image
    "erase_max_patches":  4,      # max number of patches per image

    # Grid distortion
    "grid_prob":          0.5,    # probability of applying
    "grid_steps":         4,      # number of grid cells per axis
    "grid_distort":       0.2,   # max distortion per grid point (fraction)

    # Sharpen / blur
    "sharpen_blur_prob":  0.4,    # probability of applying
    "blur_max_kernel":    3,      # max gaussian blur kernel radius (odd only)
    "sharpen_strength":   0.4,    # strength of unsharp mask
}


# ─────────────────────────────────────────────
#  SPATIAL TRANSFORMS
# ─────────────────────────────────────────────

def _rotate(img):
    h, w  = img.shape
    angle = random.uniform(-AUG_CONFIG["rotation_deg"], AUG_CONFIG["rotation_deg"])
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def _translate(img):
    h, w = img.shape
    frac = AUG_CONFIG["translate_frac"]
    tx   = random.uniform(-frac, frac) * w
    ty   = random.uniform(-frac, frac) * h
    M    = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REFLECT_101)


def _grid_distortion(img):
    """
    Simulates hand placement variation and skin deformation.
    Divides image into a grid and randomly displaces each grid point.
    Much lighter than elastic distortion but still captures position variation.
    """
    if random.random() > AUG_CONFIG["grid_prob"]:
        return img

    h, w    = img.shape
    steps   = AUG_CONFIG["grid_steps"]
    distort = AUG_CONFIG["grid_distort"]

    # Build displacement map
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    step_x = w // steps
    step_y = h // steps

    for i in range(steps + 1):
        for j in range(steps + 1):
            x = min(j * step_x, w - 1)
            y = min(i * step_y, h - 1)
            dx = random.uniform(-distort, distort) * step_x
            dy = random.uniform(-distort, distort) * step_y
            x1 = min(x + step_x, w - 1)
            y1 = min(y + step_y, h - 1)
            map_x[y:y1, x:x1] = np.linspace(x + dx, x1 + dx, x1 - x) if x1 > x else x + dx
            map_y[y:y1, x:x1] = np.linspace(y + dy, y1 + dy, y1 - y).reshape(-1, 1) if y1 > y else y + dy

    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)

    return cv2.remap(img, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


# ─────────────────────────────────────────────
#  PHOTOMETRIC TRANSFORMS
# ─────────────────────────────────────────────



def _gaussian_noise(img):
    """Mild sensor noise."""
    std   = AUG_CONFIG["noise_std"] / 255.0
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def _sharpen_blur(img):
    """
    Randomly either sharpens or blurs the image.
    Sharpening: simulates high-pressure firm hand placement (clearer veins).
    Blurring   : simulates slight motion or soft placement (softer veins).
    """
    if random.random() > AUG_CONFIG["sharpen_blur_prob"]:
        return img

    if random.random() < 0.5:
        # Blur
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    else:
        # Sharpen via unsharp mask
        strength = AUG_CONFIG["sharpen_strength"]
        blurred  = cv2.GaussianBlur(img, (5, 5), 0)
        sharp    = img + strength * (img - blurred)
        return np.clip(sharp, 0.0, 1.0)


# ─────────────────────────────────────────────
#  RANDOM ERASING
# ─────────────────────────────────────────────

def _random_erase(img):
    """
    Randomly blacks out small rectangular patches.
    Forces model not to rely on any single vein segment for identification.
    Directly addresses over-reliance on prominent veins that vary with hand position.
    """
    if random.random() > AUG_CONFIG["erase_prob"]:
        return img

    img    = img.copy()
    h, w   = img.shape
    area   = h * w
    n_patches = random.randint(1, AUG_CONFIG["erase_max_patches"])

    for _ in range(n_patches):
        erase_area = random.uniform(
            AUG_CONFIG["erase_min_frac"],
            AUG_CONFIG["erase_max_frac"]
        ) * area
        aspect = random.uniform(0.3, 3.0)
        ph     = int(np.sqrt(erase_area / aspect))
        pw     = int(np.sqrt(erase_area * aspect))
        ph     = max(1, min(ph, h - 1))
        pw     = max(1, min(pw, w - 1))
        y0     = random.randint(0, h - ph)
        x0     = random.randint(0, w - pw)
        img[y0:y0+ph, x0:x0+pw] = 0.0

    return img


# ─────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────

def augment_image(img):
    
    if img.ndim == 3:
        img = img[:, :, 0]

    # Spatial 1
    img = _rotate(img)
    img = _translate(img)

    # Photometric
    img = _sharpen_blur(img)
    img = _gaussian_noise(img)

    # Spatial 2
    img = _grid_distortion(img)
    
    # Structural
    img = _random_erase(img)

    return img[:, :, np.newaxis]