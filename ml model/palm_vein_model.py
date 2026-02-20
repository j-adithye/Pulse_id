#!/usr/bin/env python3
"""
Palm Vein Recognition - CNN + Siamese Network Training
=======================================================
Dataset structure:
    dataset/
    ├── person1/   (10 images)
    ├── person2/   (10 images)
    ...
    └── person400/ (10 images)

Pipeline:
    - Loads pairs on the fly with augmentation
    - Trains CNN backbone using Siamese + Contrastive Loss
    - Saves CNN-only model for deployment

Requirements:
    pip install tensorflow numpy opencv-python scikit-learn matplotlib
"""

import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import combinations

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    # Paths
    "dataset_dir":   "dataset",
    "output_dir":    "output",
    "backbone_path": "output/cnn_backbone.h5",   # saved after training (deployment model)
    "siamese_path":  "output/siamese_full.h5",   # full siamese (for resuming training)

    # Image
    "img_size":      224,       # final input size (matches your preprocessing output)
    "channels":      1,         # grayscale

    # Embedding
    "embedding_dim": 128,       # size of final feature vector

    # Dataset split (at CLASS level)
    "train_ratio":   0.80,      # 320 classes
    "val_ratio":     0.10,      # 40 classes
    "test_ratio":    0.10,      # 40 classes

    # Pair generation
    "neg_pos_ratio": 3,         # negative pairs per positive pair (balance)

    # Training
    "batch_size":    32,
    "epochs":        50,
    "learning_rate": 1e-4,
    "margin":        1.0,       # contrastive loss margin

    # Augmentation (safe ranges for vein images)
    "aug_rotation":     12,     # degrees ± 
    "aug_translate":    0.07,   # fraction of image size ±
    "aug_brightness":   0.15,   # ±
    "aug_contrast":     0.15,   # ±
    "aug_noise_std":    8,      # gaussian noise std (pixel values 0-255)
    "aug_flip":         True,   # horizontal flip
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ─────────────────────────────────────────────
#  AUGMENTATION  (applied on-the-fly, training only)
# ─────────────────────────────────────────────

def augment_image(img):
    """
    Apply safe augmentations that preserve vein structure.
    Input/output: float32 array, shape (H, W) or (H, W, 1), range [0, 1]
    """
    # Work in HxW float32
    if img.ndim == 3:
        img = img[:, :, 0]

    h, w = img.shape

    # 1. Random horizontal flip
    if CONFIG["aug_flip"] and random.random() < 0.5:
        img = np.fliplr(img)

    # 2. Small rotation (±12°)
    angle = random.uniform(-CONFIG["aug_rotation"], CONFIG["aug_rotation"])
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M_rot, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    # 3. Small translation (±7%)
    tx = random.uniform(-CONFIG["aug_translate"], CONFIG["aug_translate"]) * w
    ty = random.uniform(-CONFIG["aug_translate"], CONFIG["aug_translate"]) * h
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M_trans, (w, h),
                         borderMode=cv2.BORDER_REFLECT_101)

    # 4. Brightness adjustment (±15%)
    delta = random.uniform(-CONFIG["aug_brightness"], CONFIG["aug_brightness"])
    img = np.clip(img + delta, 0.0, 1.0)

    # 5. Contrast adjustment (±15%)
    factor = 1.0 + random.uniform(-CONFIG["aug_contrast"], CONFIG["aug_contrast"])
    mean = np.mean(img)
    img = np.clip((img - mean) * factor + mean, 0.0, 1.0)

    # 6. Mild Gaussian noise
    noise = np.random.normal(0, CONFIG["aug_noise_std"] / 255.0, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0.0, 1.0)

    return img[:, :, np.newaxis]   # return (H, W, 1)


# ─────────────────────────────────────────────
#  DATASET LOADER
# ─────────────────────────────────────────────

def load_dataset(dataset_dir):
    """
    Scan dataset folder and return a dict:
        { class_name: [list of image paths] }
    """
    class_map = {}
    for person in sorted(os.listdir(dataset_dir)):
        person_dir = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_dir):
            continue
        images = [
            os.path.join(person_dir, f)
            for f in sorted(os.listdir(person_dir))
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if len(images) >= 2:   # need at least 2 for a positive pair
            class_map[person] = images

    print(f"✓ Loaded {len(class_map)} classes")
    total_images = sum(len(v) for v in class_map.values())
    print(f"✓ Total images: {total_images}")
    return class_map


def split_classes(class_map):
    """Split classes into train / val / test (NO class overlap)"""
    classes = list(class_map.keys())
    random.shuffle(classes)

    n = len(classes)
    n_train = int(n * CONFIG["train_ratio"])
    n_val   = int(n * CONFIG["val_ratio"])

    train_classes = classes[:n_train]
    val_classes   = classes[n_train:n_train + n_val]
    test_classes  = classes[n_train + n_val:]

    print(f"✓ Split — Train: {len(train_classes)} | Val: {len(val_classes)} | Test: {len(test_classes)} classes")
    return (
        {c: class_map[c] for c in train_classes},
        {c: class_map[c] for c in val_classes},
        {c: class_map[c] for c in test_classes},
    )


def generate_pairs(class_map, neg_pos_ratio=None):
    """
    Generate (path_a, path_b, label) pairs.
        label = 0 → same person (positive)
        label = 1 → different person (negative)
    """
    if neg_pos_ratio is None:
        neg_pos_ratio = CONFIG["neg_pos_ratio"]

    positive_pairs = []
    for paths in class_map.values():
        for a, b in combinations(paths, 2):
            positive_pairs.append((a, b, 0))

    # Sample negatives
    class_list = list(class_map.keys())
    negative_pairs = []
    n_neg = len(positive_pairs) * neg_pos_ratio
    while len(negative_pairs) < n_neg:
        c1, c2 = random.sample(class_list, 2)
        a = random.choice(class_map[c1])
        b = random.choice(class_map[c2])
        negative_pairs.append((a, b, 1))

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    print(f"✓ Pairs — Positive: {len(positive_pairs)} | Negative: {len(negative_pairs)} | Total: {len(all_pairs)}")
    return all_pairs


def load_image(path):
    """Load and normalize a single image → float32 (H, W, 1) in [0,1]"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    img = cv2.resize(img, (CONFIG["img_size"], CONFIG["img_size"]))
    img = img.astype(np.float32) / 255.0
    return img[:, :, np.newaxis]   # (H, W, 1)


# ─────────────────────────────────────────────
#  DATA GENERATOR  (pairs + on-the-fly augmentation)
# ─────────────────────────────────────────────

class PairGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence that yields batches of image pairs on the fly.
    Augmentation is applied only when augment=True (training set).
    """

    def __init__(self, pairs, batch_size, augment=False):
        self.pairs      = pairs
        self.batch_size = batch_size
        self.augment    = augment
        self.indices    = np.arange(len(pairs))

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def on_epoch_end(self):
        """Shuffle pairs after each epoch"""
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_pairs   = [self.pairs[i] for i in batch_indices]

        img_size = CONFIG["img_size"]
        ch       = CONFIG["channels"]

        batch_a = np.zeros((len(batch_pairs), img_size, img_size, ch), dtype=np.float32)
        batch_b = np.zeros((len(batch_pairs), img_size, img_size, ch), dtype=np.float32)
        labels  = np.zeros((len(batch_pairs),), dtype=np.float32)

        for i, (path_a, path_b, label) in enumerate(batch_pairs):
            img_a = load_image(path_a)
            img_b = load_image(path_b)

            if self.augment:
                img_a = augment_image(img_a)
                img_b = augment_image(img_b)

            batch_a[i] = img_a
            batch_b[i] = img_b
            labels[i]  = label

        return [batch_a, batch_b], labels


# ─────────────────────────────────────────────
#  CNN BACKBONE  (lightweight for Raspberry Pi)
# ─────────────────────────────────────────────

def build_cnn_backbone(input_shape, embedding_dim):
    """
    Lightweight CNN backbone suitable for Raspberry Pi 4.
    Input:  (224, 224, 1)
    Output: L2-normalized embedding vector of size embedding_dim
    """
    inputs = Input(shape=input_shape, name="input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.ReLU(name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.ReLU(name="relu1_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)    # → 112×112×32
    x = layers.Dropout(0.25, name="drop1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.ReLU(name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.ReLU(name="relu2_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)    # → 56×56×64
    x = layers.Dropout(0.25, name="drop2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.ReLU(name="relu3_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.ReLU(name="relu3_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)    # → 28×28×128
    x = layers.Dropout(0.25, name="drop3")(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.ReLU(name="relu4_1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)    # → 14×14×256
    x = layers.Dropout(0.25, name="drop4")(x)

    # Global Average Pooling (much lighter than Flatten for Pi deployment)
    x = layers.GlobalAveragePooling2D(name="gap")(x)    # → 256

    # Embedding head
    x = layers.Dense(256, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.ReLU(name="relu_dense")(x)
    x = layers.Dropout(0.3, name="drop_dense")(x)
    x = layers.Dense(embedding_dim, name="embedding")(x)

    # L2 normalize → all vectors live on unit hypersphere
    # cosine similarity then = dot product
    outputs = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1),
        name="l2_norm"
    )(x)

    backbone = Model(inputs, outputs, name="cnn_backbone")
    return backbone


# ─────────────────────────────────────────────
#  SIAMESE NETWORK  (training only)
# ─────────────────────────────────────────────

def build_siamese_network(backbone):
    """
    Wrap backbone in Siamese structure for training.
    Takes two images, returns euclidean distance between embeddings.
    """
    img_size = CONFIG["img_size"]
    ch       = CONFIG["channels"]
    shape    = (img_size, img_size, ch)

    input_a = Input(shape=shape, name="input_a")
    input_b = Input(shape=shape, name="input_b")

    # Shared backbone (same weights)
    embedding_a = backbone(input_a)
    embedding_b = backbone(input_b)

    # Euclidean distance between embeddings
    distance = layers.Lambda(
        lambda tensors: tf.norm(tensors[0] - tensors[1], axis=1, keepdims=True),
        name="euclidean_distance"
    )([embedding_a, embedding_b])

    siamese = Model(inputs=[input_a, input_b], outputs=distance, name="siamese_network")
    return siamese


# ─────────────────────────────────────────────
#  CONTRASTIVE LOSS
# ─────────────────────────────────────────────

def contrastive_loss(margin=1.0):
    """
    Contrastive loss function.
        label = 0 → same person  → minimize distance
        label = 1 → diff person  → maximize distance (up to margin)

    Loss = (1-y) * D² + y * max(margin - D, 0)²
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        d      = tf.squeeze(y_pred, axis=1)
        pos    = (1.0 - y_true) * tf.square(d)
        neg    = y_true * tf.square(tf.maximum(margin - d, 0.0))
        return tf.reduce_mean(pos + neg)
    return loss


# ─────────────────────────────────────────────
#  METRICS  (accuracy at optimal threshold)
# ─────────────────────────────────────────────

def compute_accuracy(distances, labels, threshold=0.5):
    """Compute accuracy given distances and labels at a threshold"""
    predictions = (distances < threshold).astype(int)   # < threshold → same person
    true_labels = (1 - labels).astype(int)              # label 0 = same → positive
    return np.mean(predictions == true_labels)


def find_best_threshold(distances, labels, steps=100):
    """Find threshold that maximizes accuracy on validation set"""
    best_acc   = 0
    best_thresh = 0
    for t in np.linspace(0, 2, steps):
        acc = compute_accuracy(distances, labels, threshold=t)
        if acc > best_acc:
            best_acc    = acc
            best_thresh = t
    return best_thresh, best_acc


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────

def train(dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = CONFIG["dataset_dir"]

    print("\n" + "="*60)
    print("PALM VEIN CNN + SIAMESE TRAINING")
    print("="*60)

    # ── Load & split dataset ──
    class_map = load_dataset(dataset_dir)
    train_map, val_map, test_map = split_classes(class_map)

    # ── Generate pairs ──
    print("\nGenerating pairs...")
    train_pairs = generate_pairs(train_map)
    val_pairs   = generate_pairs(val_map)
    test_pairs  = generate_pairs(test_map)

    # ── Data generators ──
    train_gen = PairGenerator(train_pairs, CONFIG["batch_size"], augment=True)
    val_gen   = PairGenerator(val_pairs,   CONFIG["batch_size"], augment=False)
    test_gen  = PairGenerator(test_pairs,  CONFIG["batch_size"], augment=False)

    # ── Build models ──
    print("\nBuilding models...")
    input_shape = (CONFIG["img_size"], CONFIG["img_size"], CONFIG["channels"])
    backbone    = build_cnn_backbone(input_shape, CONFIG["embedding_dim"])
    siamese     = build_siamese_network(backbone)

    backbone.summary()

    # ── Compile ──
    siamese.compile(
        optimizer=Adam(learning_rate=CONFIG["learning_rate"]),
        loss=contrastive_loss(margin=CONFIG["margin"])
    )

    # ── Callbacks ──
    callbacks = [
        ModelCheckpoint(
            CONFIG["siamese_path"],
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    # ── Train ──
    print("\nStarting training...")
    print("="*60)
    history = siamese.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG["epochs"],
        callbacks=callbacks,
    )

    # ── Save CNN backbone only (deployment model) ──
    backbone.save(CONFIG["backbone_path"])
    print(f"\n✓ CNN backbone saved → {CONFIG['backbone_path']}")
    print("  (This is the model you deploy on Raspberry Pi)")

    # ── Evaluate on test set ──
    print("\nEvaluating on test set...")
    distances = []
    labels    = []
    for i in range(len(test_gen)):
        [batch_a, batch_b], batch_labels = test_gen[i]
        emb_a = backbone.predict(batch_a, verbose=0)
        emb_b = backbone.predict(batch_b, verbose=0)
        dist  = np.linalg.norm(emb_a - emb_b, axis=1)
        distances.extend(dist.tolist())
        labels.extend(batch_labels.tolist())

    distances = np.array(distances)
    labels    = np.array(labels)

    best_thresh, best_acc = find_best_threshold(distances, labels)
    print(f"✓ Best threshold: {best_thresh:.4f}")
    print(f"✓ Test accuracy:  {best_acc * 100:.2f}%")

    # ── Plot training curves ──
    plot_training(history)

    return backbone, history, best_thresh


# ─────────────────────────────────────────────
#  PLOT TRAINING CURVES
# ─────────────────────────────────────────────

def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Contrastive Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Learning rate (if ReduceLROnPlateau fired)
    if "lr" in history.history:
        axes[1].plot(history.history["lr"], color="orange")
        axes[1].set_title("Learning Rate")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("LR")
        axes[1].grid(True)
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plot_path = os.path.join(CONFIG["output_dir"], "training_curves.png")
    plt.savefig(plot_path)
    print(f"✓ Training curves saved → {plot_path}")
    plt.show()


# ─────────────────────────────────────────────
#  INFERENCE UTILITIES  (enrollment + auth)
# ─────────────────────────────────────────────

def load_backbone(model_path=None):
    """Load saved CNN backbone for deployment"""
    if model_path is None:
        model_path = CONFIG["backbone_path"]
    backbone = tf.keras.models.load_model(model_path)
    print(f"✓ Backbone loaded from {model_path}")
    return backbone


def enroll_user(backbone, image_paths):
    """
    Enroll a user by averaging embeddings from 5-10 images.
    Returns a single representative template vector.
    """
    embeddings = []
    for path in image_paths:
        img = load_image(path)                      # (H, W, 1)
        img = np.expand_dims(img, axis=0)           # (1, H, W, 1)
        emb = backbone.predict(img, verbose=0)[0]   # (128,)
        embeddings.append(emb)

    template = np.mean(embeddings, axis=0)
    template = template / (np.linalg.norm(template) + 1e-10)  # re-normalize
    print(f"✓ Enrolled user with {len(image_paths)} images → template shape: {template.shape}")
    return template


def authenticate_user(backbone, query_image_path, stored_template, threshold=0.5):
    """
    Authenticate a user by comparing query image embedding
    against stored template using cosine similarity.

    Returns:
        granted (bool), similarity score (float)
    """
    img = load_image(query_image_path)
    img = np.expand_dims(img, axis=0)
    query_emb = backbone.predict(img, verbose=0)[0]

    # Cosine similarity (both vectors are L2 normalized → dot product = cosine similarity)
    similarity = np.dot(query_emb, stored_template)

    # Convert to euclidean distance equivalent for threshold comparison
    # For L2-normalized vectors: euclidean_dist² = 2 * (1 - cosine_similarity)
    distance = np.sqrt(2 * max(0, 1 - similarity))

    granted = distance < threshold
    status  = "GRANTED ✓" if granted else "DENIED ✗"

    print(f"Authentication: {status}")
    print(f"  Cosine similarity: {similarity:.4f}")
    print(f"  Euclidean distance: {distance:.4f} (threshold: {threshold})")

    return granted, similarity


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("="*60)
    print("PALM VEIN RECOGNITION — MODEL TRAINING")
    print("="*60)
    print(f"Dataset:       {CONFIG['dataset_dir']}")
    print(f"Image size:    {CONFIG['img_size']}×{CONFIG['img_size']}")
    print(f"Embedding dim: {CONFIG['embedding_dim']}")
    print(f"Batch size:    {CONFIG['batch_size']}")
    print(f"Max epochs:    {CONFIG['epochs']}")
    print(f"Margin:        {CONFIG['margin']}")
    print("="*60)

    backbone, history, best_threshold = train()

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"CNN backbone saved  → {CONFIG['backbone_path']}")
    print(f"Best threshold      → {best_threshold:.4f}")
    print("\nTo deploy on Raspberry Pi:")
    print("  1. Copy cnn_backbone.h5 to your Pi")
    print("  2. Use enroll_user() to register new users")
    print("  3. Use authenticate_user() for authentication")
    print("="*60)
