#!/usr/bin/env python3
"""
ml_model/model.py
=================
Siamese CNN training for palm vein recognition.

Reads from:   ../processed_dataset/   (output of preprocessing/preprocessing.py)
Writes to:    ../output/
    cnn_backbone.h5             full Keras model  (for debugging / resuming)
    cnn_backbone.tflite         int8-quantized    (deploy this on Raspberry Pi)
    deployment_config.json      threshold + metadata consumed by auth
    training_curves.png

The images in processed_dataset/ are already uint8 PNGs (224x224 grayscale)
produced by the full preprocessing pipeline (blur → crop → wrist removal →
finger removal → segmentation → CLAHE → Sato → CLAHE → resize → z-score,
then rescaled back to 0-255 for storage).

So load_image() here just reads the PNG and divides by 255 — no
re-preprocessing, no pipeline calls.

Augmentation (rotation, translation, brightness, contrast, noise) is applied
on-the-fly during training only, imported from augmentation.py in this folder.
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")   # headless — safe for Pi and remote training servers
import matplotlib.pyplot as plt
from itertools import combinations

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# augmentation.py lives in the same folder as this file
sys.path.insert(0, os.path.dirname(__file__))
from augmentation import augment_image # type: ignore


# ─────────────────────────────────────────────
#  PATHS
#  Anchored to this file's location so the script works regardless of
#  where it's called from.
# ─────────────────────────────────────────────

_HERE         = os.path.dirname(os.path.abspath(__file__))
_ROOT         = os.path.dirname(_HERE)

PROCESSED_DIR = os.path.join(_ROOT, "processed_dataset")
OUTPUT_DIR    = os.path.join(_ROOT, "output")


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    # Paths
    "processed_dir": PROCESSED_DIR,
    "output_dir":    OUTPUT_DIR,
    "backbone_path": os.path.join(OUTPUT_DIR, "cnn_backbone.h5"),
    "siamese_path":  os.path.join(OUTPUT_DIR, "siamese_checkpoint.h5"),
    "tflite_path":   os.path.join(OUTPUT_DIR, "cnn_backbone.tflite"),
    "deploy_config": os.path.join(OUTPUT_DIR, "deployment_config.json"),
    "plot_path":     os.path.join(OUTPUT_DIR, "training_curves.png"),

    # Image — must match what preprocessing.py outputs
    "img_size":      128,
    "channels":      1,

    # Embedding
    "embedding_dim": 128,

    # Class-level split (no identity appears in more than one split)
    "train_ratio":   0.80,
    "val_ratio":     0.10,
    # test gets the remainder

    # Pair generation
    # 3:1 neg:pos for training (harder negative exposure)
    # 1:1 for val/test (unbiased accuracy metric)
    "train_neg_ratio": 3,
    "eval_neg_ratio":  1,

    # Training
    "batch_size":    32,
    "epochs":        50,
    "learning_rate": 1e-4,
    "margin":        1.0,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────

def load_dataset(processed_dir: str) -> dict:
    """
    Scan processed_dataset/ and return:
        { identity_name: [list of image paths] }

    Expects the same folder layout as the raw dataset/:
        processed_dataset/
        ├── person_001/
        │   ├── vein_1.png
        │   └── ...
        └── person_N/
    """
    class_map = {}
    for person in sorted(os.listdir(processed_dir)):
        person_dir = os.path.join(processed_dir, person)
        if not os.path.isdir(person_dir):
            continue
        images = [
            os.path.join(person_dir, f)
            for f in sorted(os.listdir(person_dir))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if len(images) >= 2:
            class_map[person] = images

    total = sum(len(v) for v in class_map.values())
    print(f"✓ Dataset loaded — {len(class_map)} identities, {total} images")
    return class_map


def split_classes(class_map: dict) -> tuple:
    """
    Random identity-level split into train / val / test.
    No identity appears in more than one split — this forces the network
    to generalise to completely unseen people rather than memorise them.
    """
    classes = list(class_map.keys())
    random.shuffle(classes)

    n       = len(classes)
    n_train = int(n * CONFIG["train_ratio"])
    n_val   = int(n * CONFIG["val_ratio"])

    train_cls = classes[:n_train]
    val_cls   = classes[n_train : n_train + n_val]
    test_cls  = classes[n_train + n_val :]

    print(f"  Split → train: {len(train_cls)}  val: {len(val_cls)}  test: {len(test_cls)} identities")
    return (
        {c: class_map[c] for c in train_cls},
        {c: class_map[c] for c in val_cls},
        {c: class_map[c] for c in test_cls},
    )


# ─────────────────────────────────────────────
#  PAIR GENERATION
# ─────────────────────────────────────────────

def generate_pairs(class_map: dict, neg_ratio: int) -> list:
    """
    Build (path_a, path_b, label) tuples.
        label = 0  →  same identity  (minimize distance)
        label = 1  →  different      (maximize distance up to margin)

    Negatives are sampled without replacement via a seen-set to prevent
    duplicate pairs from distorting the training signal.
    """
    positives = []
    for paths in class_map.values():
        for a, b in combinations(paths, 2):
            positives.append((a, b, 0))

    class_list = list(class_map.keys())
    target     = len(positives) * neg_ratio
    seen       = set()
    negatives  = []

    for _ in range(target * 10):   # 10× budget to find unique pairs
        if len(negatives) >= target:
            break
        c1, c2 = random.sample(class_list, 2)
        a = random.choice(class_map[c1])
        b = random.choice(class_map[c2])
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            negatives.append((a, b, 1))

    if len(negatives) < target:
        print(f"  ⚠  Generated {len(negatives)}/{target} unique negatives")

    pairs = positives + negatives
    random.shuffle(pairs)
    print(f"  Pairs → pos: {len(positives)}  neg: {len(negatives)}  total: {len(pairs)}")
    return pairs


# ─────────────────────────────────────────────
#  IMAGE LOADING
# ─────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """
    Load a preprocessed image from processed_dataset/.

    preprocessing.py outputs z-score normalized values, then rescales them
    back to uint8 [0, 255] for storage:
        saved = (normalized - min) / (max - min) * 255

    Dividing by 255 here restores a clean float32 [0, 1] image.
    The vein structure (enhanced by CLAHE + Sato) is fully preserved.

    Returns: float32 (H, W, 1)
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")

    # Safety resize — only triggers if img_size was changed after preprocessing
    sz = CONFIG["img_size"]
    if img.shape[0] != sz or img.shape[1] != sz:
        img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_CUBIC)

    return (img.astype(np.float32) / 255.0)[:, :, np.newaxis]   # (H, W, 1)


# ─────────────────────────────────────────────
#  DATA GENERATOR
# ─────────────────────────────────────────────

class PairGenerator(tf.keras.utils.Sequence):
    """
    Yields batches of image pairs on the fly.
    augment=True  →  stochastic augmentation via augmentation.py (training only)
    augment=False →  deterministic (val, test, threshold calibration)
    """

    def __init__(self, pairs: list, batch_size: int, augment: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pairs      = pairs
        self.batch_size = batch_size
        self.augment    = augment
        self.indices    = np.arange(len(pairs))

    def __len__(self) -> int:
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, idx: int):
        batch = [
            self.pairs[i]
            for i in self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        ]

        sz = CONFIG["img_size"]
        ch = CONFIG["channels"]
        A  = np.zeros((len(batch), sz, sz, ch), dtype=np.float32)
        B  = np.zeros((len(batch), sz, sz, ch), dtype=np.float32)
        L  = np.zeros((len(batch),),            dtype=np.float32)

        for i, (pa, pb, label) in enumerate(batch):
            ia = load_image(pa)
            ib = load_image(pb)
            if self.augment:
                ia = augment_image(ia)
                ib = augment_image(ib)
            A[i] = ia
            B[i] = ib
            L[i] = label

        return (A, B), L


# ─────────────────────────────────────────────
#  CNN BACKBONE
# ─────────────────────────────────────────────

def build_backbone(img_size: int, channels: int, embedding_dim: int) -> Model:
    """
    4-block CNN → GlobalAveragePooling → Dense head → L2-normalized embedding.

    Raspberry Pi 4 notes:
      - GlobalAveragePooling not Flatten: far fewer parameters, less RAM.
      - int8 TFLite export (done after training) gives ~4× speedup on ARMv8.
      - If inference is still too slow, drop img_size from 224 to 128 in CONFIG
        and re-run preprocessing with target_size=128 — no code changes needed.
    """
    inputs = Input(shape=(img_size, img_size, channels), name="input")

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.ReLU(name="relu1_1")(x)
    x = layers.Conv2D(32, 3, padding="same", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.ReLU(name="relu1_2")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.ReLU(name="relu2_1")(x)
    x = layers.Conv2D(64, 3, padding="same", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.ReLU(name="relu2_2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.ReLU(name="relu3_1")(x)
    x = layers.Conv2D(128, 3, padding="same", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.ReLU(name="relu3_2")(x)
    x = layers.MaxPooling2D(2, name="pool3")(x)
    x = layers.Dropout(0.25, name="drop3")(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding="same", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.ReLU(name="relu4_1")(x)
    x = layers.MaxPooling2D(2, name="pool4")(x)
    x = layers.Dropout(0.25, name="drop4")(x)

    # Pooling + embedding head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.ReLU(name="relu_dense")(x)
    x = layers.Dropout(0.3, name="drop_dense")(x)
    x = layers.Dense(embedding_dim, name="embedding")(x)

    # L2 normalize → unit hypersphere, cosine similarity == dot product
    outputs = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1),
        name="l2_norm"
    )(x)

    return Model(inputs, outputs, name="cnn_backbone")


# ─────────────────────────────────────────────
#  SIAMESE WRAPPER  (training only)
# ─────────────────────────────────────────────

def build_siamese(backbone: Model) -> Model:
    """Wrap backbone for pair-wise training. Only the backbone is deployed."""
    shape = (CONFIG["img_size"], CONFIG["img_size"], CONFIG["channels"])
    in_a  = Input(shape=shape, name="input_a")
    in_b  = Input(shape=shape, name="input_b")

    emb_a = backbone(in_a, training=True)
    emb_b = backbone(in_b, training=True)

    dist = layers.Lambda(
        lambda t: tf.norm(t[0] - t[1], axis=1, keepdims=True),
        name="euclidean_dist"
    )([emb_a, emb_b])

    return Model(inputs=[in_a, in_b], outputs=dist, name="siamese")


# ─────────────────────────────────────────────
#  CONTRASTIVE LOSS
# ─────────────────────────────────────────────

def contrastive_loss(margin: float = 1.0):
    """
    L = (1-y)·d²  +  y·max(margin-d, 0)²
    y=0 → same person → push d toward 0
    y=1 → different   → push d beyond margin
    """
    def loss(y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        d = tf.squeeze(y_pred, axis=1)
        return tf.reduce_mean(
            (1.0 - y) * tf.square(d)
            + y * tf.square(tf.maximum(margin - d, 0.0))
        )
    return loss


# ─────────────────────────────────────────────
#  EVALUATION HELPERS
# ─────────────────────────────────────────────

def get_distances(backbone: Model, generator: PairGenerator) -> tuple:
    """Run backbone (inference mode) over a generator, return distances + labels."""
    all_dist, all_labels = [], []
    for i in range(len(generator)):
        [A, B], labels = generator[i]
        emb_a = backbone(A, training=False).numpy()
        emb_b = backbone(B, training=False).numpy()
        all_dist.extend(np.linalg.norm(emb_a - emb_b, axis=1).tolist())
        all_labels.extend(labels.tolist())
    return np.array(all_dist), np.array(all_labels)


def accuracy_at_threshold(dists: np.ndarray, labels: np.ndarray, t: float) -> float:
    return float(np.mean((dists < t).astype(int) == (1 - labels).astype(int)))


def find_best_threshold(dists: np.ndarray, labels: np.ndarray, steps: int = 200) -> tuple:
    """Sweep distance thresholds, return the one with highest accuracy."""
    best_t, best_acc = 0.0, 0.0
    for t in np.linspace(0, 2, steps):
        acc = accuracy_at_threshold(dists, labels, t)
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


# ─────────────────────────────────────────────
#  TFLITE EXPORT
# ─────────────────────────────────────────────

def export_tflite(backbone: Model, train_map: dict, tflite_path: str):
    """
    Convert the trained backbone to int8-quantized TFLite.

    Why int8:
      - ~4× faster inference on ARMv8 (Raspberry Pi 4)
      - ~4× smaller model file
      - Minimal accuracy loss (<0.5% typical for this type of network)

    The representative_dataset feeds ~300 real training images to the
    converter so it can calibrate the int8 activation ranges correctly.
    Without this the converter falls back to float32, defeating the purpose.

    IO tensors stay float32 — auth.py doesn't need to handle quantized IO.
    """
    print("\nExporting to TFLite (int8 quantization)...")

    sample_paths = []
    for paths in train_map.values():
        sample_paths.extend(paths)
    random.shuffle(sample_paths)
    sample_paths = sample_paths[:300]

    def representative_dataset():
        for path in sample_paths:
            img = load_image(path)                      # (H, W, 1) float32
            yield [np.expand_dims(img, axis=0)]         # (1, H, W, 1)

    converter = tf.lite.TFLiteConverter.from_keras_model(backbone)
    converter.optimizations             = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset    = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type      = tf.float32   # keep float IO
    converter.inference_output_type     = tf.float32

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"✓ TFLite model → {tflite_path}  ({size_kb:.0f} KB)")


# ─────────────────────────────────────────────
#  TRAINING CURVES
# ─────────────────────────────────────────────

def save_training_plot(history, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("Contrastive Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    if "lr" in history.history:
        axes[1].plot(history.history["lr"], color="orange")
        axes[1].set_title("Learning Rate")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(True)
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"✓ Training curves → {path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def train(processed_dir: str = None):
    if processed_dir is None:
        processed_dir = CONFIG["processed_dir"]

    print("\n" + "=" * 60)
    print("PALM VEIN — SIAMESE CNN TRAINING")
    print("=" * 60)
    print(f"  Source : {processed_dir}")
    print(f"  Output : {CONFIG['output_dir']}")
    print(f"  Size   : {CONFIG['img_size']}×{CONFIG['img_size']}")
    print(f"  Embed  : {CONFIG['embedding_dim']}-d")
    print("=" * 60)

    # ── Load & split ──
    class_map = load_dataset(processed_dir)
    train_map, val_map, test_map = split_classes(class_map)

    # ── Pairs ──
    print("\nGenerating pairs...")
    train_pairs = generate_pairs(train_map, CONFIG["train_neg_ratio"])
    val_pairs   = generate_pairs(val_map,   CONFIG["eval_neg_ratio"])
    test_pairs  = generate_pairs(test_map,  CONFIG["eval_neg_ratio"])

    # ── Generators ──
    train_gen = PairGenerator(train_pairs, CONFIG["batch_size"], augment=True)
    val_gen   = PairGenerator(val_pairs,   CONFIG["batch_size"], augment=False)
    test_gen  = PairGenerator(test_pairs,  CONFIG["batch_size"], augment=False)

    # ── Build ──
    print("\nBuilding model...")
    backbone = build_backbone(CONFIG["img_size"], CONFIG["channels"], CONFIG["embedding_dim"])
    siamese  = build_siamese(backbone)
    backbone.summary()

    siamese.compile(
        optimizer=Adam(learning_rate=CONFIG["learning_rate"]),
        loss=contrastive_loss(margin=CONFIG["margin"])
    )

    # ── Callbacks ──
    callbacks = [
        ModelCheckpoint(CONFIG["siamese_path"], monitor="val_loss",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
    ]

    # ── Train ──
    print("\nTraining...")
    history = siamese.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG["epochs"],
        callbacks=callbacks,
    )

    # ── Save Keras backbone ──
    backbone.save(CONFIG["backbone_path"])
    print(f"✓ Backbone (Keras) → {CONFIG['backbone_path']}")

    # ── Threshold: calibrate on VAL, report on TEST ──
    # Calibrating on test would be data leakage — the threshold would be
    # optimistically tuned to that specific split.
    print("\nCalibrating threshold on validation set...")
    val_dists, val_labels = get_distances(backbone, val_gen)
    best_thresh, val_acc  = find_best_threshold(val_dists, val_labels)
    print(f"  Val accuracy  : {val_acc * 100:.2f}%  at threshold = {best_thresh:.4f}")

    print("Evaluating on test set...")
    test_dists, test_labels = get_distances(backbone, test_gen)
    test_acc = accuracy_at_threshold(test_dists, test_labels, best_thresh)
    print(f"  Test accuracy : {test_acc * 100:.2f}%  at threshold = {best_thresh:.4f}")

    # ── TFLite ──
    export_tflite(backbone, train_map, CONFIG["tflite_path"])

    # ── Deployment config ──
    deploy_cfg = {
        "tflite_path":   CONFIG["tflite_path"],
        "backbone_path": CONFIG["backbone_path"],
        "threshold":     best_thresh,
        "embedding_dim": CONFIG["embedding_dim"],
        "img_size":      CONFIG["img_size"],
    }
    with open(CONFIG["deploy_config"], "w") as f:
        json.dump(deploy_cfg, f, indent=2)
    print(f"✓ Deployment config → {CONFIG['deploy_config']}")

    # ── Plot ──
    save_training_plot(history, CONFIG["plot_path"])

    return backbone, history, best_thresh


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Palm Vein CNN")
    parser.add_argument(
        "--data", default=None,
        help=f"Path to processed_dataset/  (default: {PROCESSED_DIR})"
    )
    args = parser.parse_args()

    backbone, history, threshold = train(processed_dir=args.data)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  TFLite model   → {CONFIG['tflite_path']}")
    print(f"  Deployment cfg → {CONFIG['deploy_config']}")
    print(f"  Threshold      → {threshold:.4f}")
    print("=" * 60)