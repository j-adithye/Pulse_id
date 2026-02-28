#!/usr/bin/env python3
"""
ml_model/model.py
=================
Siamese CNN training for palm vein recognition — with Hard Negative Mining.

Reads from:   ../processed_dataset/
Writes to:    ../output/
    cnn_backbone.h5
    cnn_backbone.tflite         (deploy on Raspberry Pi)
    deployment_config.json
    training_curves.png
    training_log.csv

HARD NEGATIVE MINING
--------------------
Random negatives are trivially easy — the model separates them at epoch 0
and stops learning. Hard negative mining fixes this by:

  1. After each epoch, running all training images through the backbone
  2. Finding negative pairs where embeddings are CLOSE (distance < margin)
     — these are the cases the model is currently getting wrong
  3. Replacing the training pairs with these hard pairs for the next epoch

The result: train loss actually decreases meaningfully instead of
flatlining at epoch 0.

FINE-TUNING
-----------
Pass --finetune to load the existing cnn_backbone.h5 and continue training.
Recommended — the current model already has reasonable embeddings to mine from.

    python ml_model/palm_vein_model.py --finetune

Scratch training (default):
    python ml_model/palm_vein_model.py
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)

sys.path.insert(0, os.path.dirname(__file__))
from augmentation import augment_image  # type: ignore


# ─────────────────────────────────────────────
#  PATHS
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
    "log_path":      os.path.join(OUTPUT_DIR, "training_log.csv"),

    # Image
    "img_size":      224,
    "channels":      1,

    # Embedding
    "embedding_dim": 128,

    # Class split
    "train_ratio":   0.80,
    "val_ratio":     0.10,

    # Pair generation
    "train_neg_ratio": 3,
    "eval_neg_ratio":  1,

    # Hard negative mining
    # 0.7 = 70% hard negatives, 30% random — keeps variety in training
    "hard_neg_fraction": 0.7,

    # Training
    "batch_size":    32,
    "epochs":        50,
    "learning_rate": 1e-4,
    "finetune_lr":   1e-5,   # lower LR when fine-tuning existing model
    "margin":        1.0,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────

def load_dataset(processed_dir: str) -> dict:
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
    Initial pair generation with random negatives.
    Used for val/test throughout, and for epoch 0 of training.
    Hard mining takes over from epoch 1 onward.
    """
    positives = []
    for paths in class_map.values():
        for a, b in combinations(paths, 2):
            positives.append((a, b, 0))

    class_list = list(class_map.keys())
    target     = len(positives) * neg_ratio
    seen       = set()
    negatives  = []

    for _ in range(target * 10):
        if len(negatives) >= target:
            break
        c1, c2 = random.sample(class_list, 2)
        a = random.choice(class_map[c1])
        b = random.choice(class_map[c2])
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            negatives.append((a, b, 1))

    pairs = positives + negatives
    random.shuffle(pairs)
    print(f"  Pairs → pos: {len(positives)}  neg: {len(negatives)}  total: {len(pairs)}")
    return pairs


def mine_hard_negatives(backbone: Model, class_map: dict,
                        n_positives: int) -> list:
    """
    Find hard negative pairs using current backbone embeddings.

    Strategy:
      1. Embed all training images (inference mode, no augmentation)
      2. For each anchor image, find close embeddings from OTHER identities
         within the margin — these are confusable pairs the model struggles with
      3. Mix 70% hard + 30% random negatives to maintain variety

    Returns list of (path_a, path_b, label=1) negative pairs.
    """
    print("  Mining hard negatives...")

    # ── Embed all training images ──
    all_paths      = []
    all_identities = []
    for identity, paths in class_map.items():
        for p in paths:
            all_paths.append(p)
            all_identities.append(identity)

    all_identities = np.array(all_identities)
    sz = CONFIG["img_size"]
    ch = CONFIG["channels"]

    batch_size = 64
    embeddings = []
    for i in range(0, len(all_paths), batch_size):
        batch_paths = all_paths[i : i + batch_size]
        imgs = np.zeros((len(batch_paths), sz, sz, ch), dtype=np.float32)
        for j, p in enumerate(batch_paths):
            imgs[j] = load_image(p)
        embs = backbone(imgs, training=False).numpy()
        embeddings.append(embs)

    embeddings = np.vstack(embeddings)   # (N, 128)

    # ── Mine hard negatives ──
    margin       = CONFIG["margin"]
    target       = n_positives * CONFIG["train_neg_ratio"]
    n_hard       = int(target * CONFIG["hard_neg_fraction"])
    n_random     = target - n_hard

    unique_ids   = list(class_map.keys())
    mine_sample  = min(50, len(unique_ids))

    hard_negatives = []
    seen           = set()
    attempts       = 0

    while len(hard_negatives) < n_hard and attempts < n_hard * 20:
        attempts += 1

        id_a  = random.choice(unique_ids)
        idx_a = random.choice(np.where(all_identities == id_a)[0])
        emb_a = embeddings[idx_a]

        other_ids  = [i for i in unique_ids if i != id_a]
        sample_ids = random.sample(other_ids, min(mine_sample, len(other_ids)))

        for id_b in sample_ids:
            idxs_b = np.where(all_identities == id_b)[0]
            embs_b = embeddings[idxs_b]
            dists  = np.linalg.norm(embs_b - emb_a, axis=1)

            # Hard = within margin (model is confused about these)
            hard_idxs = np.where(dists < margin)[0]
            if len(hard_idxs) == 0:
                continue

            hardest = hard_idxs[np.argmin(dists[hard_idxs])]
            path_a  = all_paths[idx_a]
            path_b  = all_paths[idxs_b[hardest]]
            key     = (min(path_a, path_b), max(path_a, path_b))

            if key not in seen:
                seen.add(key)
                hard_negatives.append((path_a, path_b, 1))
                break

    # ── Fill remainder with random negatives ──
    class_list       = list(class_map.keys())
    random_negatives = []
    for _ in range(n_random * 5):
        if len(random_negatives) >= n_random:
            break
        c1, c2 = random.sample(class_list, 2)
        a = random.choice(class_map[c1])
        b = random.choice(class_map[c2])
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            random_negatives.append((a, b, 1))

    all_negatives = hard_negatives + random_negatives
    print(f"  Hard: {len(hard_negatives)}  Random: {len(random_negatives)}  "
          f"Total negatives: {len(all_negatives)}")
    return all_negatives


# ─────────────────────────────────────────────
#  IMAGE LOADING
# ─────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load preprocessed image. Returns float32 (H, W, 1) in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    sz = CONFIG["img_size"]
    if img.shape[0] != sz or img.shape[1] != sz:
        img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_CUBIC)
    return (img.astype(np.float32) / 255.0)[:, :, np.newaxis]


# ─────────────────────────────────────────────
#  DATA GENERATOR
# ─────────────────────────────────────────────

class PairGenerator(tf.keras.utils.Sequence):
    """
    Yields batches of image pairs.
    Pairs are refreshed each epoch via update_pairs() called by
    HardNegativeMiningCallback.
    """

    def __init__(self, pairs: list, batch_size: int,
                 augment: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pairs      = pairs
        self.batch_size = batch_size
        self.augment    = augment
        self.indices    = np.arange(len(pairs))

    def update_pairs(self, new_pairs: list):
        """Replace pairs in-place. Called by hard mining callback."""
        self.pairs   = new_pairs
        self.indices = np.arange(len(new_pairs))
        np.random.shuffle(self.indices)

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
#  HARD NEGATIVE MINING CALLBACK
# ─────────────────────────────────────────────

class HardNegativeMiningCallback(tf.keras.callbacks.Callback):
    """
    Refreshes training pairs with hard negatives after each epoch.
    Skips epoch 0 — embeddings need at least one epoch to be meaningful.
    """

    def __init__(self, backbone: Model, train_gen: PairGenerator,
                 class_map: dict, n_positives: int):
        super().__init__()
        self.backbone    = backbone
        self.train_gen   = train_gen
        self.class_map   = class_map
        self.n_positives = n_positives

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            return

        print(f"\n[Epoch {epoch + 1}] Refreshing pairs with hard negatives...")

        positives      = [p for p in self.train_gen.pairs if p[2] == 0]
        hard_negatives = mine_hard_negatives(
            self.backbone, self.class_map, self.n_positives
        )

        new_pairs = positives + hard_negatives
        random.shuffle(new_pairs)
        self.train_gen.update_pairs(new_pairs)
        print(f"  Training pairs updated: {len(new_pairs)} total")


# ─────────────────────────────────────────────
#  CNN BACKBONE
# ─────────────────────────────────────────────

def build_backbone(img_size: int, channels: int, embedding_dim: int) -> Model:
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

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.ReLU(name="relu_dense")(x)
    x = layers.Dropout(0.3, name="drop_dense")(x)
    x = layers.Dense(embedding_dim, name="embedding")(x)

    outputs = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1),
        name="l2_norm"
    )(x)

    return Model(inputs, outputs, name="cnn_backbone")


# ─────────────────────────────────────────────
#  SIAMESE WRAPPER
# ─────────────────────────────────────────────

def build_siamese(backbone: Model) -> Model:
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
    def loss(y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        d = tf.squeeze(y_pred, axis=1)
        return tf.reduce_mean(
            (1.0 - y) * tf.square(d)
            + y * tf.square(tf.maximum(margin - d, 0.0))
        )
    return loss


# ─────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────

def get_distances(backbone: Model, generator: PairGenerator) -> tuple:
    all_dist, all_labels = [], []
    for i in range(len(generator)):
        (A, B), labels = generator[i]
        emb_a = backbone(A, training=False).numpy()
        emb_b = backbone(B, training=False).numpy()
        all_dist.extend(np.linalg.norm(emb_a - emb_b, axis=1).tolist())
        all_labels.extend(labels.tolist())
    return np.array(all_dist), np.array(all_labels)


def accuracy_at_threshold(dists: np.ndarray, labels: np.ndarray,
                           t: float) -> float:
    return float(np.mean((dists < t).astype(int) == (1 - labels).astype(int)))


def find_best_threshold(dists: np.ndarray, labels: np.ndarray,
                        steps: int = 200) -> tuple:
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
    print("\nExporting to TFLite (int8 quantization)...")

    sample_paths = []
    for paths in train_map.values():
        sample_paths.extend(paths)
    random.shuffle(sample_paths)
    sample_paths = sample_paths[:300]

    def representative_dataset():
        for path in sample_paths:
            img = load_image(path)
            yield [np.expand_dims(img, axis=0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(backbone)
    converter.optimizations             = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset    = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type      = tf.float32
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

def train(processed_dir: str = None, finetune: bool = False):
    if processed_dir is None:
        processed_dir = CONFIG["processed_dir"]

    mode = "FINE-TUNE" if finetune else "SCRATCH"
    print("\n" + "=" * 60)
    print(f"PALM VEIN — SIAMESE CNN TRAINING  [{mode} + Hard Negative Mining]")
    print("=" * 60)
    print(f"  Source           : {processed_dir}")
    print(f"  Output           : {CONFIG['output_dir']}")
    print(f"  Size             : {CONFIG['img_size']}x{CONFIG['img_size']}")
    print(f"  Embed            : {CONFIG['embedding_dim']}-d")
    print(f"  Hard neg fraction: {CONFIG['hard_neg_fraction']}")
    print("=" * 60)

    # ── Load & split ──
    class_map = load_dataset(processed_dir)
    train_map, val_map, test_map = split_classes(class_map)

    # ── Initial pairs (random negatives — used for epoch 0) ──
    print("\nGenerating initial pairs...")
    train_pairs = generate_pairs(train_map, CONFIG["train_neg_ratio"])
    val_pairs   = generate_pairs(val_map,   CONFIG["eval_neg_ratio"])
    test_pairs  = generate_pairs(test_map,  CONFIG["eval_neg_ratio"])

    n_positives = sum(1 for p in train_pairs if p[2] == 0)

    # ── Generators ──
    train_gen = PairGenerator(train_pairs, CONFIG["batch_size"], augment=True)
    val_gen   = PairGenerator(val_pairs,   CONFIG["batch_size"], augment=False)
    test_gen  = PairGenerator(test_pairs,  CONFIG["batch_size"], augment=False)

    # ── Build or load backbone ──
    print("\nBuilding model...")
    if finetune and os.path.exists(CONFIG["backbone_path"]):
        print(f"  Loading existing backbone: {CONFIG['backbone_path']}")
        backbone = tf.keras.models.load_model(CONFIG["backbone_path"])
        lr = CONFIG["finetune_lr"]
        print(f"  Fine-tune LR: {lr}")
    else:
        if finetune:
            print("  WARNING: No existing backbone found — training from scratch")
        backbone = build_backbone(
            CONFIG["img_size"], CONFIG["channels"], CONFIG["embedding_dim"]
        )
        lr = CONFIG["learning_rate"]

    siamese = build_siamese(backbone)
    backbone.summary()

    siamese.compile(
        optimizer=Adam(learning_rate=lr),
        loss=contrastive_loss(margin=CONFIG["margin"])
    )

    # ── Callbacks ──
    hard_mining_cb = HardNegativeMiningCallback(
        backbone    = backbone,
        train_gen   = train_gen,
        class_map   = train_map,
        n_positives = n_positives,
    )

    callbacks = [
        hard_mining_cb,
        ModelCheckpoint(CONFIG["siamese_path"], monitor="val_loss",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
        CSVLogger(CONFIG["log_path"], append=finetune),
    ]

    # ── Train ──
    print("\nTraining...")
    history = siamese.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG["epochs"],
        callbacks=callbacks,
    )

    # ── Save backbone ──
    backbone.save(CONFIG["backbone_path"])
    print(f"✓ Backbone (Keras) → {CONFIG['backbone_path']}")

    # ── Threshold on val, evaluate on test ──
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
        help=f"Path to processed_dataset/ (default: {PROCESSED_DIR})"
    )
    parser.add_argument(
        "--finetune", action="store_true",
        help="Load existing cnn_backbone.h5 and fine-tune with hard negatives"
    )
    args = parser.parse_args()

    backbone, history, threshold = train(
        processed_dir=args.data,
        finetune=args.finetune
    )

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  TFLite model   → {CONFIG['tflite_path']}")
    print(f"  Deployment cfg → {CONFIG['deploy_config']}")
    print(f"  Threshold      → {threshold:.4f}")
    print("=" * 60)
