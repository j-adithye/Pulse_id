"""
config.py - Central configuration for palm vein auth system

All tunable parameters in one place.
Import this in any file instead of hardcoding values.
"""

import os
import json


# -- Paths -------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "cnn_backbone.tflite")
DEPLOY_CFG  = os.path.join(BASE_DIR, "models", "deployment_config.json")
DB_PATH     = os.path.join(BASE_DIR, "data", "embeddings.db")


# -- Camera ------------------------------------------------------------------
AWB_GAINS     = (1.0, 1.0)   # (red_gain, blue_gain) fixed, no auto white balance
SHUTTER_SPEED = 19000         # microseconds
ANALOGUE_GAIN = 1.1
CONTRAST      = 1.6
CAPTURE_SIZE  = (1296, 972)   # OV5647 native 4:3 resolution


# -- Model -------------------------------------------------------------------
IMG_SIZE      = 128           # must match training
EMBEDDING_DIM = 128            # must match training

THRESHOLD = 0.325


# -- Preprocessing -----------------------------------------------------------
OTSU_OFFSET          = -20
WRIST_JUNCTION_RISE  = 0.15
MCP_SAFETY_MARGIN    = 0.03
MCP_DEFECT_DEPTH_MIN = 20     # pixels
SATO_SCALE_MIN       = 3
SATO_SCALE_MAX       = 3
SATO_SCALE_STEP      = 1
CLAHE_PRE_CLIP       = 1.5
CLAHE_POST_CLIP      = 0.5
CLAHE_TILE_SIZE      = 8


# -- Liveness (MLX90614) -----------------------------------------------------
I2C_BUS             = 1       # Pi I2C bus number
MLX90614_ADDR       = 0x5A    # default I2C address
HAND_TEMP_MIN       = 30.0    # degrees Celsius
HAND_TEMP_MAX       = 37.0    # degrees Celsius


# -- Flask -------------------------------------------------------------------
HOST  = "0.0.0.0"             # listen on all interfaces so LAN devices can connect
PORT  = 5000
DEBUG = False
