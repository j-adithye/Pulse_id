#!/usr/bin/env python3
"""
augmentation.py
===============
Handles all image augmentation for palm vein recognition.

Augmentations are intentionally conservative — they must preserve
vein structure while providing enough variation to prevent overfitting.

NOTE: Horizontal flip is deliberately excluded. Left/right hand vein
patterns are NOT symmetric; flipping a left-hand image produces
something resembling a right hand, which poisons the embedding space.

Usage (called by model.py during training):
    from augmentation import augment_image
    augmented = augment_image(img)   # float32 (H, W, 1) → (H, W, 1)
"""

import random
import numpy as np
import cv2


# ─────────────────────────────────────────────
#  AUGMENTATION CONFIG
#  Keep these ranges conservative for vein images.
#  Aggressive augmentation destroys the fine vein structure.
# ─────────────────────────────────────────────

AUG_CONFIG = {
    "rotation_deg":   12,       # ± degrees  — small enough to stay within sensor FOV
    "translate_frac": 0.07,     # ± fraction of image size — mimics small hand shifts
    "brightness":     0.15,     # ± additive on [0,1] — accounts for IR lighting variance
    "contrast":       0.15,     # ± multiplicative factor around mean
    "noise_std":      8,        # gaussian noise std in pixel space (0–255 scale)
                                # divided by 255 internally — keeps veins visible
}


# ─────────────────────────────────────────────
#  INDIVIDUAL TRANSFORMS
#  Each function takes and returns float32 (H, W) in [0, 1].
#  Keeping them separate makes it easy to ablate or reorder.
# ─────────────────────────────────────────────

def _rotate(img: np.ndarray) -> np.ndarray:
    """Random small rotation. BORDER_REFLECT_101 avoids black border artifacts."""
    h, w = img.shape
    angle = random.uniform(-AUG_CONFIG["rotation_deg"], AUG_CONFIG["rotation_deg"])
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def _translate(img: np.ndarray) -> np.ndarray:
    """Random small translation. Mimics slight repositioning of hand on sensor."""
    h, w = img.shape
    frac = AUG_CONFIG["translate_frac"]
    tx = random.uniform(-frac, frac) * w
    ty = random.uniform(-frac, frac) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REFLECT_101)


def _brightness(img: np.ndarray) -> np.ndarray:
    """Additive brightness shift. Simulates IR illumination variation."""
    delta = random.uniform(-AUG_CONFIG["brightness"], AUG_CONFIG["brightness"])
    return np.clip(img + delta, 0.0, 1.0)


def _contrast(img: np.ndarray) -> np.ndarray:
    """Contrast adjustment around the image mean. Preserves relative vein structure."""
    factor = 1.0 + random.uniform(-AUG_CONFIG["contrast"], AUG_CONFIG["contrast"])
    mean = np.mean(img)
    return np.clip((img - mean) * factor + mean, 0.0, 1.0)


def _gaussian_noise(img: np.ndarray) -> np.ndarray:
    """Mild Gaussian noise. Prevents over-reliance on exact pixel values."""
    std = AUG_CONFIG["noise_std"] / 255.0
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


# ─────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────

def augment_image(img: np.ndarray) -> np.ndarray:
    """
    Apply the full augmentation pipeline to a single vein image.

    Args:
        img: float32 array, shape (H, W, 1) or (H, W), values in [0, 1].
             Typically the output of load_image() in model.py.

    Returns:
        float32 array, shape (H, W, 1), values in [0, 1].

    Pipeline order matters:
        1. Spatial transforms first (rotation, translation)  — so noise isn't
           interpolated across borders
        2. Photometric transforms second (brightness, contrast, noise)
    """
    # Normalise to 2D for OpenCV operations
    if img.ndim == 3:
        img = img[:, :, 0]

    img = _rotate(img)
    img = _translate(img)
    img = _brightness(img)
    img = _contrast(img)
    img = _gaussian_noise(img)

    return img[:, :, np.newaxis]   # restore channel dim → (H, W, 1)
