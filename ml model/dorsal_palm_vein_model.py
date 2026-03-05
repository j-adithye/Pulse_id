#!/usr/bin/env python3
"""
Palm Vein CNN — Triplet Loss + Online Semi-Hard Mining
Changes from v2 (contrastive + offline mining):
  - Triplet loss replacing contrastive loss
  - Online semi-hard mining inside each batch (no separate mining callback)
  - Identity-based batch sampler: K identities x N images per batch
  - Val loss is real triplet loss on val batches
  - All architecture improvements from v2 kept:
      conv4_2, embedding_dim=128, no conv dropout, cosine annealing LR
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
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, CSVLogger
)

sys.path.insert(0, '/kaggle/working')
from augmentation import augment_image


# ── PATHS ──
OUTPUT_DIR = '/kaggle/working/output'
CKPT_DIR   = '/kaggle/working/checkpoints'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)


# ── CONFIG ──
CONFIG = {
    'output_dir':    OUTPUT_DIR,
    'backbone_path': os.path.join(OUTPUT_DIR, 'cnn_backbone.h5'),
    'best_ckpt':     os.path.join(OUTPUT_DIR, 'backbone_best.h5'),
    'tflite_path':   os.path.join(OUTPUT_DIR, 'cnn_backbone.tflite'),
    'deploy_config': os.path.join(OUTPUT_DIR, 'deployment_config.json'),
    'plot_path':     os.path.join(OUTPUT_DIR, 'training_curves.png'),
    'log_path':      os.path.join(OUTPUT_DIR, 'training_log.csv'),
    'ckpt_dir':      CKPT_DIR,

    'img_size':      128,
    'channels':      1,
    'embedding_dim': 128,

    'train_ratio':   0.75,
    'val_ratio':     0.15,

    # Identity sampler — K identities x N images = batch size
    'K':             32,    # identities per batch
    'N':             4,     # images per identity per batch

    'epochs':        50,
    'learning_rate': 1e-4,
    'min_lr':        1e-6,
    'es_patience':   15,

    'margin':        1.5,

    'warmup_epochs': 5,
    'restart_every': 20,
}


# ────────────────────────────────
#  DATASET
# ────────────────────────────────

def load_dataset(processed_dir):
    class_map = {}
    for person in sorted(os.listdir(processed_dir)):
        person_dir = os.path.join(processed_dir, person)
        if not os.path.isdir(person_dir):
            continue
        images = [
            os.path.join(person_dir, f)
            for f in sorted(os.listdir(person_dir))
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if len(images) >= CONFIG['N']:
            class_map[person] = images
    total = sum(len(v) for v in class_map.values())
    print(f'Dataset: {len(class_map)} identities, {total} images')
    return class_map


def split_classes(class_map):
    classes = list(class_map.keys())
    random.shuffle(classes)
    n         = len(classes)
    n_train   = int(n * CONFIG['train_ratio'])
    n_val     = int(n * CONFIG['val_ratio'])
    train_cls = classes[:n_train]
    val_cls   = classes[n_train : n_train + n_val]
    test_cls  = classes[n_train + n_val :]
    print(f'Split: train={len(train_cls)}  val={len(val_cls)}  test={len(test_cls)}')
    return (
        {c: class_map[c] for c in train_cls},
        {c: class_map[c] for c in val_cls},
        {c: class_map[c] for c in test_cls},
    )


# ────────────────────────────────
#  IMAGE LOADING
# ────────────────────────────────

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Cannot load: {path}')
    sz = CONFIG['img_size']
    if img.shape[0] != sz or img.shape[1] != sz:
        img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_CUBIC)
    return (img.astype(np.float32) / 255.0)[:, :, np.newaxis]


# ────────────────────────────────
#  IDENTITY BATCH SAMPLER
# ────────────────────────────────

class IdentityBatchGenerator(tf.keras.utils.Sequence):
    """
    Samples K identities per batch, N images per identity.
    Each batch contains K*N images with known identity labels.
    Online semi-hard mining happens inside the triplet loss.
    """
    def __init__(self, class_map, K, N, augment=False, steps_per_epoch=None, **kwargs):
        super().__init__(**kwargs)
        self.class_map       = class_map
        self.K               = K
        self.N               = N
        self.augment         = augment
        self.identities      = list(class_map.keys())
        self.steps_per_epoch = steps_per_epoch or len(self.identities) // K

    def __len__(self):
        return self.steps_per_epoch

    def on_epoch_end(self):
        random.shuffle(self.identities)

    def __getitem__(self, idx):
        sz = CONFIG['img_size']
        ch = CONFIG['channels']
        K  = self.K
        N  = self.N

        # Sample K identities
        selected_ids = random.sample(self.identities, min(K, len(self.identities)))

        imgs   = np.zeros((K * N, sz, sz, ch), dtype=np.float32)
        labels = np.zeros((K * N,),            dtype=np.int32)

        for i, identity in enumerate(selected_ids):
            paths = self.class_map[identity]
            # Sample N images from this identity (with replacement if needed)
            sampled = random.choices(paths, k=N) if len(paths) < N else random.sample(paths, N)
            for j, path in enumerate(sampled):
                img = load_image(path)
                if self.augment:
                    img = augment_image(img)
                imgs[i * N + j]   = img
                labels[i * N + j] = i   # label = index within batch (0 to K-1)

        return imgs, labels


# ────────────────────────────────
#  BACKBONE
# ────────────────────────────────

@register_keras_serializable(package='PalmVein')
class L2NormalizeLayer(layers.Layer):
    def call(self, x):
        return tf.math.l2_normalize(x, axis=1)
    def get_config(self):
        return super().get_config()


def build_backbone(img_size, channels, embedding_dim):
    inputs = Input(shape=(img_size, img_size, channels), name='input')

    # Block 1
    x = layers.Conv2D(32, 3, padding='same', name='conv1_1')(inputs)
    x = layers.BatchNormalization(name='bn1_1')(x)
    x = layers.ReLU(name='relu1_1')(x)
    x = layers.Conv2D(32, 3, padding='same', name='conv1_2')(x)
    x = layers.BatchNormalization(name='bn1_2')(x)
    x = layers.ReLU(name='relu1_2')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding='same', name='conv2_1')(x)
    x = layers.BatchNormalization(name='bn2_1')(x)
    x = layers.ReLU(name='relu2_1')(x)
    x = layers.Conv2D(64, 3, padding='same', name='conv2_2')(x)
    x = layers.BatchNormalization(name='bn2_2')(x)
    x = layers.ReLU(name='relu2_2')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding='same', name='conv3_1')(x)
    x = layers.BatchNormalization(name='bn3_1')(x)
    x = layers.ReLU(name='relu3_1')(x)
    x = layers.Conv2D(128, 3, padding='same', name='conv3_2')(x)
    x = layers.BatchNormalization(name='bn3_2')(x)
    x = layers.ReLU(name='relu3_2')(x)
    x = layers.MaxPooling2D(2, name='pool3')(x)

    # Block 4 — symmetric
    x = layers.Conv2D(256, 3, padding='same', name='conv4_1')(x)
    x = layers.BatchNormalization(name='bn4_1')(x)
    x = layers.ReLU(name='relu4_1')(x)
    x = layers.Conv2D(256, 3, padding='same', name='conv4_2')(x)
    x = layers.BatchNormalization(name='bn4_2')(x)
    x = layers.ReLU(name='relu4_2')(x)
    x = layers.MaxPooling2D(2, name='pool4')(x)

    # Head
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(256, name='dense1')(x)
    x = layers.BatchNormalization(name='bn_dense')(x)
    x = layers.ReLU(name='relu_dense')(x)
    x = layers.Dropout(0.2, name='drop_dense')(x)
    x = layers.Dense(embedding_dim, name='embedding')(x)
    outputs = L2NormalizeLayer(name='l2_norm')(x)

    return Model(inputs, outputs, name='cnn_backbone')


# ────────────────────────────────
#  ONLINE SEMI-HARD TRIPLET LOSS
# ────────────────────────────────

def online_semi_hard_triplet_loss(margin=1.5):
    def loss(y_true, y_pred):
        embeddings = tf.cast(y_pred, tf.float32)
        labels     = tf.cast(tf.squeeze(y_true), tf.int32)

        # Pairwise distance matrix
        dot        = tf.matmul(embeddings, tf.transpose(embeddings))
        sq_norm    = tf.linalg.diag_part(dot)
        distances  = tf.maximum(sq_norm[:, None] - 2.0 * dot + sq_norm[None, :], 0.0)
        distances  = tf.sqrt(distances + 1e-12)

        # Masks — all ops vectorized, no Python loop
        labels_col = tf.expand_dims(labels, 1)
        labels_row = tf.expand_dims(labels, 0)
        same_id    = tf.equal(labels_col, labels_row)
        diff_id    = tf.logical_not(same_id)
        not_self   = tf.logical_not(tf.eye(tf.shape(labels)[0], dtype=tf.bool))
        pos_mask   = tf.logical_and(same_id, not_self)
        neg_mask   = diff_id

        # Hardest positive per anchor — same identity, max distance
        pos_dist   = tf.where(pos_mask, distances, tf.zeros_like(distances))
        d_pos      = tf.reduce_max(pos_dist, axis=1, keepdims=True)  # (batch, 1)

        # Semi-hard negatives — diff identity where d_pos < d_neg < d_pos + margin
        neg_dist       = tf.where(neg_mask, distances, tf.fill(tf.shape(distances), 1e9))
        semi_hard_mask = tf.logical_and(
            neg_dist > d_pos,
            neg_dist < d_pos + margin
        )

        # Closest semi-hard negative per anchor
        semi_hard_dist = tf.where(semi_hard_mask, neg_dist, tf.fill(tf.shape(neg_dist), 1e9))
        d_neg_semi     = tf.reduce_min(semi_hard_dist, axis=1)

        # Fallback — hardest negative (for anchors with no semi-hard)
        d_neg_hard     = tf.reduce_min(neg_dist, axis=1)

        # Use semi-hard if exists, else hardest negative
        has_semi_hard  = tf.reduce_any(semi_hard_mask, axis=1)
        d_neg          = tf.where(has_semi_hard, d_neg_semi, d_neg_hard)

        triplet_loss   = tf.maximum(tf.squeeze(d_pos) - d_neg + margin, 0.0)
        return tf.reduce_mean(triplet_loss)

    return loss


# ────────────────────────────────
#  CALLBACKS
# ────────────────────────────────

class BackboneCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, backbone, ckpt_dir):
        super().__init__()
        self.backbone = backbone
        self.ckpt_dir = ckpt_dir

    def on_epoch_end(self, epoch, logs=None):
        path     = os.path.join(self.ckpt_dir, f'backbone_epoch_{epoch+1:02d}.h5')
        val_loss = logs.get('val_loss', float('nan'))
        self.backbone.save(path)
        print(f'  Backbone checkpoint saved: backbone_epoch_{epoch+1:02d}.h5'
              f'  (val_loss={val_loss:.4f})')


class BestBackboneCheckpoint(tf.keras.callbacks.Callback):
    """Saves backbone (not the wrapper model) when val_loss improves."""
    def __init__(self, backbone, path):
        super().__init__()
        self.backbone  = backbone
        self.path      = path
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss', float('inf'))
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.backbone.save(self.path)
            print(f'  Best backbone saved  (val_loss={val_loss:.4f})')


class CosineAnnealingLR(tf.keras.callbacks.Callback):
    def __init__(self, max_lr, min_lr, epochs, warmup_epochs, restart_every):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.restart_every = restart_every

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (epoch / self.warmup_epochs)
        else:
            cycle_epoch = (epoch - self.warmup_epochs) % self.restart_every
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + np.cos(np.pi * cycle_epoch / self.restart_every)
            )
        self.model.optimizer.learning_rate.assign(lr)


# ────────────────────────────────
#  WRAPPER MODEL
# ────────────────────────────────

def build_wrapper(backbone):
    """
    Wraps backbone so model.fit() works with identity batch generator.
    Input: batch of images
    Output: embeddings (loss computed by online_semi_hard_triplet_loss)
    """
    sz     = CONFIG['img_size']
    ch     = CONFIG['channels']
    inputs = Input(shape=(sz, sz, ch), name='input')
    embs   = backbone(inputs)
    return Model(inputs=inputs, outputs=embs, name='wrapper')


# ────────────────────────────────
#  EVALUATION
# ────────────────────────────────

def get_pair_distances(backbone, class_map, n_pairs=5000):
    """Generate genuine/impostor pairs and compute distances for evaluation."""
    class_list = list(class_map.keys())
    pairs      = []

    for identity, paths in class_map.items():
        for a, b in combinations(paths, 2):
            pairs.append((a, b, 0))

    neg_target = min(len(pairs), n_pairs // 2)
    seen       = set()
    neg_count  = 0
    for _ in range(neg_target * 10):
        if neg_count >= neg_target:
            break
        c1, c2 = random.sample(class_list, 2)
        a = random.choice(class_map[c1])
        b = random.choice(class_map[c2])
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            pairs.append((a, b, 1))
            neg_count += 1

    random.shuffle(pairs)
    if len(pairs) > n_pairs:
        pairs = random.sample(pairs, n_pairs)

    sz = CONFIG['img_size']
    ch = CONFIG['channels']
    bs = 128

    all_dist, all_labels = [], []
    for i in range(0, len(pairs), bs):
        batch = pairs[i : i + bs]
        A = np.zeros((len(batch), sz, sz, ch), dtype=np.float32)
        B = np.zeros((len(batch), sz, sz, ch), dtype=np.float32)
        L = []
        for j, (pa, pb, label) in enumerate(batch):
            A[j] = load_image(pa)
            B[j] = load_image(pb)
            L.append(label)
        emb_a = backbone(A, training=False).numpy()
        emb_b = backbone(B, training=False).numpy()
        dists = np.linalg.norm(emb_a - emb_b, axis=1)
        all_dist.extend(dists.tolist())
        all_labels.extend(L)

    return np.array(all_dist), np.array(all_labels)


def accuracy_at_threshold(dists, labels, t):
    return float(np.mean((dists < t).astype(int) == (1 - labels).astype(int)))


def find_best_threshold(dists, labels, steps=500):
    best_t, best_acc = 0.0, 0.0
    for t in np.linspace(0, 2, steps):
        acc = accuracy_at_threshold(dists, labels, t)
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def compute_far_frr_eer(dists, labels, steps=500):
    thresholds = np.linspace(0, 2, steps)
    genuine    = dists[labels == 0]
    impostor   = dists[labels == 1]
    far_list, frr_list = [], []
    for t in thresholds:
        far_list.append(float(np.mean(impostor < t)))
        frr_list.append(float(np.mean(genuine  >= t)))
    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    diff    = np.abs(far_arr - frr_arr)
    eer_idx = np.argmin(diff)
    eer     = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2)
    return {
        'thresholds': thresholds,
        'far':        far_arr,
        'frr':        frr_arr,
        'eer':        eer,
        'eer_thresh': float(thresholds[eer_idx]),
        'far_at_eer': float(far_arr[eer_idx]),
        'frr_at_eer': float(frr_arr[eer_idx]),
    }


# ────────────────────────────────
#  TFLITE EXPORT
# ────────────────────────────────

def export_tflite(backbone, train_map, tflite_path):
    print('\nExporting to TFLite (int8)...')
    sample_paths = []
    for paths in train_map.values():
        sample_paths.extend(paths)
    random.shuffle(sample_paths)
    sample_paths = sample_paths[:300]

    def representative_dataset():
        for path in sample_paths:
            yield [np.expand_dims(load_image(path), axis=0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(backbone)
    converter.optimizations             = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset    = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type      = tf.float32
    converter.inference_output_type     = tf.float32
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f'TFLite saved: {tflite_path}  ({os.path.getsize(tflite_path)//1024} KB)')


# ────────────────────────────────
#  TRAINING CURVES
# ────────────────────────────────

def save_training_plot(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'],     label='Train')
    axes[0].plot(history.history['val_loss'], label='Val')
    axes[0].set_title('Triplet Loss (Online Semi-Hard Mining)')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f'Training curves saved: {path}')


# ────────────────────────────────
#  MAIN
# ────────────────────────────────

def train(processed_dir):
    print('=' * 60)
    print('PALM VEIN TRAINING  [Triplet Loss + Online Semi-Hard Mining]')
    print('=' * 60)
    print(f'  Data         : {processed_dir}')
    print(f'  Output       : {OUTPUT_DIR}')
    print(f'  Size         : {CONFIG["img_size"]}x{CONFIG["img_size"]}')
    print(f'  Embedding dim: {CONFIG["embedding_dim"]}')
    print(f'  Margin       : {CONFIG["margin"]}')
    print(f'  Batch        : K={CONFIG["K"]} identities x N={CONFIG["N"]} images = {CONFIG["K"]*CONFIG["N"]}')
    print('=' * 60)

    class_map = load_dataset(processed_dir)
    train_map, val_map, test_map = split_classes(class_map)

    steps_per_epoch = len(train_map) // CONFIG['K'] * 8  # ~3 passes per epoch
    val_steps       = max(10, len(val_map) // CONFIG['K'])

    print(f'\n  Steps/epoch  : {steps_per_epoch}')
    print(f'  Val steps    : {val_steps}')

    train_gen = IdentityBatchGenerator(
        train_map, CONFIG['K'], CONFIG['N'],
        augment=True, steps_per_epoch=steps_per_epoch
    )
    val_gen = IdentityBatchGenerator(
        val_map, CONFIG['K'], CONFIG['N'],
        augment=False, steps_per_epoch=val_steps
    )

    print('\nBuilding model...')
    backbone = build_backbone(CONFIG['img_size'], CONFIG['channels'], CONFIG['embedding_dim'])
    wrapper  = build_wrapper(backbone)

    trainable = sum(np.prod(v.shape) for v in backbone.trainable_weights)
    print(f'  Backbone params: {trainable:,}')

    wrapper.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        loss=online_semi_hard_triplet_loss(margin=CONFIG['margin'])
    )

    print(f'\n  LR          : {CONFIG["learning_rate"]} (cosine anneal -> {CONFIG["min_lr"]})')
    print(f'  Max epochs  : {CONFIG["epochs"]}')
    print(f'  ES patience : {CONFIG["es_patience"]}')

    callbacks = [
        BestBackboneCheckpoint(backbone, CONFIG['best_ckpt']),
        BackboneCheckpoint(backbone, CONFIG['ckpt_dir']),
        CosineAnnealingLR(
            max_lr=CONFIG['learning_rate'],
            min_lr=CONFIG['min_lr'],
            epochs=CONFIG['epochs'],
            warmup_epochs=CONFIG['warmup_epochs'],
            restart_every=CONFIG['restart_every'],
        ),
        EarlyStopping(monitor='val_loss', patience=CONFIG['es_patience'],
                      restore_best_weights=True, verbose=1),
        CSVLogger(CONFIG['log_path']),
    ]

    print('\nTraining...')
    history = wrapper.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1,
    )

    # Load best backbone for evaluation
    best_backbone = tf.keras.models.load_model(
        CONFIG['best_ckpt'],
        custom_objects={'L2NormalizeLayer': L2NormalizeLayer}
    )
    best_backbone.save(CONFIG['backbone_path'])
    print(f'Best backbone saved: {CONFIG["backbone_path"]}')

    print('\nCalibrating threshold on validation set...')
    val_dists, val_labels = get_pair_distances(best_backbone, val_map)
    best_thresh, val_acc  = find_best_threshold(val_dists, val_labels)
    print(f'  Val accuracy  : {val_acc*100:.2f}%  threshold={best_thresh:.4f}')

    print('\nEvaluating on test set...')
    test_dists, test_labels = get_pair_distances(best_backbone, test_map)
    test_acc = accuracy_at_threshold(test_dists, test_labels, best_thresh)
    metrics  = compute_far_frr_eer(test_dists, test_labels)

    print(f'\n{"="*60}')
    print(f'  TEST RESULTS')
    print(f'{"="*60}')
    print(f'  Accuracy  : {test_acc*100:.2f}%  (threshold={best_thresh:.4f})')
    print(f'  EER       : {metrics["eer"]*100:.2f}%  (threshold={metrics["eer_thresh"]:.4f})')
    print(f'  FAR@EER   : {metrics["far_at_eer"]*100:.2f}%')
    print(f'  FRR@EER   : {metrics["frr_at_eer"]*100:.2f}%')
    print(f'{"="*60}')

    export_tflite(best_backbone, train_map, CONFIG['tflite_path'])

    deploy_cfg = {
        'tflite_path':   CONFIG['tflite_path'],
        'backbone_path': CONFIG['backbone_path'],
        'threshold':     metrics['eer_thresh'],
        'embedding_dim': CONFIG['embedding_dim'],
        'img_size':      CONFIG['img_size'],
        'eer':           metrics['eer'],
        'far_at_eer':    metrics['far_at_eer'],
        'frr_at_eer':    metrics['frr_at_eer'],
    }
    with open(CONFIG['deploy_config'], 'w') as f:
        json.dump(deploy_cfg, f, indent=2)
    print(f'Deployment config saved: {CONFIG["deploy_config"]}')

    save_training_plot(history, CONFIG['plot_path'])
    return best_backbone, history, best_thresh, metrics


# ────────────────────────────────
#  CLI
# ────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to processed_dataset/')
    args = parser.parse_args()

    backbone, history, threshold, metrics = train(processed_dir=args.data)

    print('\n' + '=' * 60)
    print('DONE')
    print(f'  TFLite    : {CONFIG["tflite_path"]}')
    print(f'  Config    : {CONFIG["deploy_config"]}')
    print(f'  Threshold : {threshold:.4f}')
    print(f'  EER       : {metrics["eer"]*100:.2f}%')
    print('=' * 60)