"""
inference.py - TFLite model loading and embedding extraction
Loads the model once at startup.
Applies full preprocessing pipeline before inference.
"""

import numpy as np
import tensorflow as tf
tflite = tf.lite
import config
from preprocessing import (
    apply_gaussian_blur,
    smart_crop_hand_region,
    remove_wrist_geometric,
    remove_fingers_mcp,
    segment_hand_otsu,
    apply_clahe,
    apply_sato_filter,
    apply_feathered_mask,
    pad_and_resize,
    normalize_image,
)

_interpreter = None
_input_idx   = None
_output_idx  = None


def _load_model():
    global _interpreter, _input_idx, _output_idx
    _interpreter = tflite.Interpreter(model_path=config.MODEL_PATH)
    _interpreter.allocate_tensors()
    _input_idx  = _interpreter.get_input_details()[0]["index"]
    _output_idx = _interpreter.get_output_details()[0]["index"]
    print("[inference] Model loaded: " + config.MODEL_PATH)


def _preprocess(raw_gray):
    """
    Run full preprocessing pipeline on raw grayscale image.
    Input : uint8 numpy array (H, W) from camera.capture()
    Output: float32 numpy array (IMG_SIZE, IMG_SIZE) normalized
    """
    img = apply_gaussian_blur(raw_gray, kernel_size=11)

    img, _ = smart_crop_hand_region(
        img,
        target_size=(512, 512),
        padding_percent=0.15,
        otsu_offset=config.OTSU_OFFSET,
    )

    img, wrist_col, wrist_side = remove_wrist_geometric(
        img,
        otsu_offset=config.OTSU_OFFSET,
        junction_rise=config.WRIST_JUNCTION_RISE,
    )

    img, _ = remove_fingers_mcp(
        img,
        wrist_cut_col=wrist_col,
        wrist_side=wrist_side,
        otsu_offset=config.OTSU_OFFSET,
        safety_margin=config.MCP_SAFETY_MARGIN,
        defect_depth_min=config.MCP_DEFECT_DEPTH_MIN,
    )

    mask, _ = segment_hand_otsu(img, otsu_offset=config.OTSU_OFFSET)

    img = apply_clahe(img, clip_limit=config.CLAHE_PRE_CLIP,
                      tile_size=config.CLAHE_TILE_SIZE, label="pre")

    img = apply_sato_filter(
        img,
        scale_min=config.SATO_SCALE_MIN,
        scale_max=config.SATO_SCALE_MAX,
        scale_step=config.SATO_SCALE_STEP,
        black_ridges=True,
    )

    img = apply_feathered_mask(img, mask, fade_px=12)
    img = apply_clahe(img, clip_limit=config.CLAHE_POST_CLIP,
                      tile_size=config.CLAHE_TILE_SIZE, label="post")
    img = pad_and_resize(img, mask, target_size=config.IMG_SIZE)
    img = normalize_image(img)
    return img


def get_embedding(raw_gray):
    """
    Preprocess a raw camera image and return its 64-dim embedding.

    Args:
        raw_gray : uint8 numpy array (H, W) from camera.capture()

    Returns:
        embedding : float32 numpy array shape (64,)
    """
    global _interpreter
    if _interpreter is None:
        _load_model()

    processed = _preprocess(raw_gray)
    tensor = processed.astype(np.float32)[np.newaxis, :, :, np.newaxis]

    _interpreter.set_tensor(_input_idx, tensor)
    _interpreter.invoke()
    embedding = _interpreter.get_tensor(_output_idx)[0]

    return embedding.copy()
