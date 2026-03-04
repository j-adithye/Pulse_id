#!/usr/bin/env python3
"""
augmentation.py
Conservative augmentation for palm vein images.
NOTE: Horizontal flip deliberately excluded — left/right vein patterns
are NOT symmetric. Flipping poisons the embedding space.
"""
import random
import numpy as np
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter

AUG_CONFIG = {
    "rotation_deg":     12,
    "translate_frac":   0.07,
    "noise_std":        8,
    # Elastic distortion: controls how strongly the grid is deformed.
    # alpha = displacement magnitude, sigma = smoothing (keep high to avoid tears)
    "elastic_alpha":    12.0,
    "elastic_sigma":    4.0,
    # Gamma: range around 1.0 — values <1 brighten, >1 darken non-linearly
    "gamma_range":      (0.75, 1.35),
}


def _rotate(img):
    h, w = img.shape
    angle = random.uniform(-AUG_CONFIG["rotation_deg"], AUG_CONFIG["rotation_deg"])
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def _translate(img):
    h, w = img.shape
    frac = AUG_CONFIG["translate_frac"]
    tx = random.uniform(-frac, frac) * w
    ty = random.uniform(-frac, frac) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _gaussian_noise(img):
    std = AUG_CONFIG["noise_std"] / 255.0
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)

def _gamma_correction(img):
    """
    Applies random gamma correction to simulate NIR illumination variance.
    Unlike linear brightness scaling, gamma acts non-linearly — it compresses
    or expands dynamic range differently in shadows vs highlights, which better
    matches real sensor response variation.

    gamma < 1 → lifts shadows (brighter overall)
    gamma > 1 → deepens shadows (darker overall)
    """
    gamma = random.uniform(*AUG_CONFIG["gamma_range"])
    # Avoid log(0); img is already in [0, 1]
    return np.power(np.clip(img, 1e-6, 1.0), gamma).astype(np.float32)


def augment_image(img):
    """
    Args: float32 (H, W, 1) in [0, 1]
    Returns: float32 (H, W, 1) in [0, 1]
    """
    if img.ndim == 3:
        img = img[:, :, 0]

    img = _rotate(img)
    img = _translate(img)
    img = _gamma_correction(img)
    img = _gaussian_noise(img)

    return img[:, :, np.newaxis]