#!/usr/bin/env python3
"""
Dorsal Palm Vein Siamese CNN — Kaggle Training Script
Hard Negative Mining + per-epoch backbone checkpoint
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
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)

sys.path.insert(0, '/kaggle/working')
from augmentation import augment_image


# ── PATHS ──
OUTPUT_DIR    = '/kaggle/working/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CHECKPOINT DIR — backbone saved every epoch ──
CKPT_DIR = '/kaggle/working/checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)


# ── CONFIG ──
CONFIG = {
    'output_dir':    OUTPUT_DIR,
    'backbone_path': os.path.join(OUTPUT_DIR, 'cnn_backbone.h5'),
    'finetune_load_path': '/kaggle/input/models/suban1234/cnn-backbone/keras/default/1/cnn_backbone.h5',
    'siamese_ckpt':  os.path.join(OUTPUT_DIR, 'siamese_best.h5'),
    'tflite_path':   os.path.join(OUTPUT_DIR, 'cnn_backbone.tflite'),
    'deploy_config': os.path.join(OUTPUT_DIR, 'deployment_config.json'),
    'plot_path':     os.path.join(OUTPUT_DIR, 'training_curves.png'),
    'log_path':      os.path.join(OUTPUT_DIR, 'training_log.csv'),
    'ckpt_dir':      CKPT_DIR,

    'img_size':          128,
    'channels':          1,
    'embedding_dim':     64,

    'train_ratio':       0.75,
    'val_ratio':         0.15,

    'train_neg_ratio':   3,
    'eval_neg_ratio':    1,
    'hard_neg_fraction': 0.6,

    'batch_size':        128,
    'epochs':            50,
    'learning_rate':     1e-4,

    # ── Fine-tune specific ──
    'finetune_lr':          5e-5,  # warm restart — higher than 1e-5, lower than 1e-4
    'finetune_epochs':      25,    # max epochs — model is already converged
    'finetune_es_patience': 7,     # tighter early stopping
    'finetune_lr_patience': 3,     # faster LR reduction

    'margin':            1.5,
}

# Layers to FREEZE during fine-tuning (conv1 + conv2 blocks)
# Only conv3, conv4, dense, embedding will be updated
FINETUNE_FREEZE = [
    'conv1_1', 'bn1_1', 'relu1_1',
    'conv1_2', 'bn1_2', 'relu1_2', 'pool1', 'drop1',
    'conv2_1', 'bn2_1', 'relu2_1',
    'conv2_2', 'bn2_2', 'relu2_2', 'pool2', 'drop2',
]


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
        if len(images) >= 2:
            class_map[person] = images
    total = sum(len(v) for v in class_map.values())
    print(f'Dataset: {len(class_map)} identities, {total} images')
    return class_map


def split_classes(class_map):
    classes = list(class_map.keys())
    random.shuffle(classes)
    n       = len(classes)
    n_train = int(n * CONFIG['train_ratio'])
    n_val   = int(n * CONFIG['val_ratio'])
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
#  PAIR GENERATION
# ────────────────────────────────

def generate_pairs(class_map, neg_ratio):
    positives = []
    for paths in class_map.values():
        for a, b in combinations(paths, 2):
            positives.append((a, b, 0))

    class_list = list(class_map.keys())
    target     = int(len(positives) * neg_ratio)
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
    print(f'  Pairs: pos={len(positives)}  neg={len(negatives)}  total={len(pairs)}')
    return pairs


def mine_hard_negatives(backbone, class_map, n_positives):
    print('  Mining hard negatives...')

    all_paths, all_ids = [], []
    for identity, paths in class_map.items():
        for p in paths:
            all_paths.append(p)
            all_ids.append(identity)
    all_ids = np.array(all_ids)

    sz = CONFIG['img_size']
    ch = CONFIG['channels']

    embeddings = []
    for i in range(0, len(all_paths), CONFIG['batch_size']):
        batch = all_paths[i : i + CONFIG['batch_size']]
        imgs  = np.zeros((len(batch), sz, sz, ch), dtype=np.float32)
        for j, p in enumerate(batch):
            imgs[j] = load_image(p)
        embeddings.append(backbone(imgs, training=False).numpy())
    embeddings = np.vstack(embeddings)

    margin     = CONFIG['margin']
    target     = int(n_positives * CONFIG['train_neg_ratio'])
    n_hard_max = int(target * CONFIG['hard_neg_fraction'])  # ceiling, not target
    n_random_min = target - n_hard_max                      # minimum random floor
    unique_ids = list(class_map.keys())
    mine_n     = min(100, len(unique_ids))   # increased from 50 → 100

    id_to_indices = {uid: np.where(all_ids == uid)[0] for uid in unique_ids}

    hard_negs = []
    seen      = set()
    attempts  = 0

    while len(hard_negs) < n_hard_max and attempts < n_hard_max * 20:
        attempts += 1
        id_a  = random.choice(unique_ids)
        idx_a = random.choice(id_to_indices[id_a])
        emb_a = embeddings[idx_a]

        other = [i for i in unique_ids if i != id_a]
        for id_b in random.sample(other, min(mine_n, len(other))):
            idxs_b = id_to_indices[id_b]
            dists  = np.linalg.norm(embeddings[idxs_b] - emb_a, axis=1)
            hard   = np.where(dists < margin)[0]
            if len(hard) == 0:
                continue
            h      = hard[np.argmin(dists[hard])]
            pa, pb = all_paths[idx_a], all_paths[idxs_b[h]]
            key    = (min(pa, pb), max(pa, pb))
            if key not in seen:
                seen.add(key)
                hard_negs.append((pa, pb, 1))
                break

    # Random fills remaining up to target, but at least n_random_min
    n_random_actual = max(n_random_min, target - len(hard_negs))
    rand_negs = []
    cl = list(class_map.keys())
    for _ in range(n_random_actual * 5):
        if len(rand_negs) >= n_random_actual:
            break
        c1, c2 = random.sample(cl, 2)
        a, b   = random.choice(class_map[c1]), random.choice(class_map[c2])
        key    = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            rand_negs.append((a, b, 1))

    all_negs = hard_negs + rand_negs
    print(f'  Hard={len(hard_negs)}  Random={len(rand_negs)}  Total={len(all_negs)}')
    return all_negs


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
#  DATA GENERATOR
# ────────────────────────────────

class PairGenerator(tf.keras.utils.Sequence):
    def __init__(self, pairs, batch_size, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.pairs      = pairs
        self.batch_size = batch_size
        self.augment    = augment
        self.indices    = np.arange(len(pairs))

    def update_pairs(self, new_pairs):
        self.pairs   = new_pairs
        self.indices = np.arange(len(new_pairs))
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch = [self.pairs[i]
                 for i in self.indices[idx*self.batch_size : (idx+1)*self.batch_size]]
        sz = CONFIG['img_size']
        ch = CONFIG['channels']
        A  = np.zeros((len(batch), sz, sz, ch), dtype=np.float32)
        B  = np.zeros((len(batch), sz, sz, ch), dtype=np.float32)
        L  = np.zeros((len(batch),),            dtype=np.float32)
        for i, (pa, pb, label) in enumerate(batch):
            ia, ib = load_image(pa), load_image(pb)
            if self.augment:
                ia, ib = augment_image(ia), augment_image(ib)
            A[i], B[i], L[i] = ia, ib, label
        return (A, B), L


# ────────────────────────────────
#  CALLBACKS
# ────────────────────────────────

class HardNegativeMiningCallback(tf.keras.callbacks.Callback):
    """Refresh training pairs with hard negatives after each epoch."""
    def __init__(self, backbone, train_gen, class_map, n_positives):
        super().__init__()
        self.backbone    = backbone
        self.train_gen   = train_gen
        self.class_map   = class_map
        self.n_positives = n_positives

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            return
        print(f'\n[Epoch {epoch+1}] Refreshing pairs with hard negatives...')
        positives = [p for p in self.train_gen.pairs if p[2] == 0]
        hard_negs = mine_hard_negatives(self.backbone, self.class_map, self.n_positives)
        new_pairs = positives + hard_negs
        random.shuffle(new_pairs)
        self.train_gen.update_pairs(new_pairs)
        print(f'  Pairs updated: {len(new_pairs)} total')


class BackboneCheckpoint(tf.keras.callbacks.Callback):
    """Saves the backbone (not the siamese wrapper) after every epoch."""
    def __init__(self, backbone, ckpt_dir):
        super().__init__()
        self.backbone = backbone
        self.ckpt_dir = ckpt_dir

    def on_epoch_end(self, epoch, logs=None):
        path = os.path.join(self.ckpt_dir, f'backbone_epoch_{epoch+1:02d}.h5')
        self.backbone.save(path)
        val_loss = logs.get('val_loss', float('nan'))
        print(f'  Backbone checkpoint saved: backbone_epoch_{epoch+1:02d}.h5  '
              f'(val_loss={val_loss:.4f})')


# ────────────────────────────────
#  CNN BACKBONE
# ────────────────────────────────

@register_keras_serializable(package='Dorsal PalmVein')
class L2NormalizeLayer(layers.Layer):
    """L2-normalizes embeddings along axis=1."""
    def call(self, x):
        return tf.math.l2_normalize(x, axis=1)
    def get_config(self):
        return super().get_config()


@register_keras_serializable(package='Dorsal PalmVein')
class EuclideanDistanceLayer(layers.Layer):
    """Euclidean distance between two embedding vectors."""
    def call(self, inputs):
        a, b = inputs
        return tf.norm(a - b, axis=1, keepdims=True)
    def get_config(self):
        return super().get_config()


def build_backbone(img_size, channels, embedding_dim):
    inputs = Input(shape=(img_size, img_size, channels), name='input')

    x = layers.Conv2D(32, 3, padding='same', name='conv1_1')(inputs)
    x = layers.BatchNormalization(name='bn1_1')(x)
    x = layers.ReLU(name='relu1_1')(x)
    x = layers.Conv2D(32, 3, padding='same', name='conv1_2')(x)
    x = layers.BatchNormalization(name='bn1_2')(x)
    x = layers.ReLU(name='relu1_2')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)
    x = layers.Dropout(0.15, name='drop1')(x)

    x = layers.Conv2D(64, 3, padding='same', name='conv2_1')(x)
    x = layers.BatchNormalization(name='bn2_1')(x)
    x = layers.ReLU(name='relu2_1')(x)
    x = layers.Conv2D(64, 3, padding='same', name='conv2_2')(x)
    x = layers.BatchNormalization(name='bn2_2')(x)
    x = layers.ReLU(name='relu2_2')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)
    x = layers.Dropout(0.15, name='drop2')(x)

    x = layers.Conv2D(128, 3, padding='same', name='conv3_1')(x)
    x = layers.BatchNormalization(name='bn3_1')(x)
    x = layers.ReLU(name='relu3_1')(x)
    x = layers.Conv2D(128, 3, padding='same', name='conv3_2')(x)
    x = layers.BatchNormalization(name='bn3_2')(x)
    x = layers.ReLU(name='relu3_2')(x)
    x = layers.MaxPooling2D(2, name='pool3')(x)
    x = layers.Dropout(0.15, name='drop3')(x)

    x = layers.Conv2D(256, 3, padding='same', name='conv4_1')(x)
    x = layers.BatchNormalization(name='bn4_1')(x)
    x = layers.ReLU(name='relu4_1')(x)
    x = layers.MaxPooling2D(2, name='pool4')(x)
    x = layers.Dropout(0.15, name='drop4')(x)

    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(256, name='dense1')(x)
    x = layers.BatchNormalization(name='bn_dense')(x)
    x = layers.ReLU(name='relu_dense')(x)
    x = layers.Dropout(0.2, name='drop_dense')(x)
    x = layers.Dense(embedding_dim, name='embedding')(x)
    outputs = L2NormalizeLayer(name='l2_norm')(x)
    return Model(inputs, outputs, name='cnn_backbone')


def build_siamese(backbone):
    shape = (CONFIG['img_size'], CONFIG['img_size'], CONFIG['channels'])
    in_a  = Input(shape=shape, name='input_a')
    in_b  = Input(shape=shape, name='input_b')
    emb_a = backbone(in_a)
    emb_b = backbone(in_b)
    dist  = EuclideanDistanceLayer(name='euclidean_dist')([emb_a, emb_b])
    return Model(inputs=[in_a, in_b], outputs=dist, name='siamese')


def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        d = tf.squeeze(y_pred, axis=1)
        return tf.reduce_mean(
            (1.0 - y) * tf.square(d)
            + y * tf.square(tf.maximum(margin - d, 0.0))
        )
    return loss


# ────────────────────────────────
#  EVALUATION
# ────────────────────────────────

def get_distances(backbone, generator):
    all_dist, all_labels = [], []
    for i in range(len(generator)):
        (A, B), labels = generator[i]
        emb_a = backbone(A, training=False).numpy()
        emb_b = backbone(B, training=False).numpy()
        all_dist.extend(np.linalg.norm(emb_a - emb_b, axis=1).tolist())
        all_labels.extend(labels.tolist())
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

    far_arr  = np.array(far_list)
    frr_arr  = np.array(frr_list)
    diff     = np.abs(far_arr - frr_arr)
    eer_idx  = np.argmin(diff)
    eer      = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2)

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
    axes[0].set_title('Contrastive Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)
    if 'lr' in history.history:
        axes[1].plot(history.history['lr'], color='orange')
        axes[1].set_title('Learning Rate')
        axes[1].set_xlabel('Epoch')
        axes[1].grid(True)
    else:
        axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f'Training curves saved: {path}')


# ────────────────────────────────
#  MAIN TRAIN FUNCTION
# ────────────────────────────────

def train(processed_dir, finetune=False):
    mode = 'FINE-TUNE' if finetune else 'SCRATCH'
    print('=' * 60)
    print(f'Dorsal Palm VEIN TRAINING  [{mode} + Hard Negative Mining]')
    print('=' * 60)
    print(f'  Data   : {processed_dir}')
    print(f'  Output : {CONFIG["output_dir"]}')
    print(f'  Size   : {CONFIG["img_size"]}x{CONFIG["img_size"]}')
    print(f'  Checkpoints: {CONFIG["ckpt_dir"]}')
    print('=' * 60)

    class_map = load_dataset(processed_dir)
    train_map, val_map, test_map = split_classes(class_map)

    print('\nGenerating initial pairs...')
    train_pairs = generate_pairs(train_map, CONFIG['train_neg_ratio'])
    val_pairs   = generate_pairs(val_map,   CONFIG['eval_neg_ratio'])
    test_pairs  = generate_pairs(test_map,  CONFIG['eval_neg_ratio'])
    n_positives = sum(1 for p in train_pairs if p[2] == 0)

    train_gen = PairGenerator(train_pairs, CONFIG['batch_size'], augment=True)
    val_gen   = PairGenerator(val_pairs,   CONFIG['batch_size'], augment=False)
    test_gen  = PairGenerator(test_pairs,  CONFIG['batch_size'], augment=False)

    print('\nBuilding model...')
    if finetune and os.path.exists(CONFIG['finetune_load_path']):
        print(f'  Loading backbone for fine-tuning: {CONFIG["backbone_path"]}')
        backbone = tf.keras.models.load_model(
            CONFIG['finetune_load_path'],
            custom_objects={
                'L2NormalizeLayer':       L2NormalizeLayer,
                'EuclideanDistanceLayer': EuclideanDistanceLayer,
            }
        )

        # Freeze conv1 + conv2 — only fine-tune conv3, conv4, dense, embedding
        for layer in backbone.layers:
            layer.trainable = layer.name not in FINETUNE_FREEZE

        frozen    = [l.name for l in backbone.layers if not l.trainable]
        trainable = [l.name for l in backbone.layers if l.trainable]
        print(f'  Frozen   : {frozen}')
        print(f'  Trainable: {trainable}')

        lr          = CONFIG['finetune_lr']
        epochs      = CONFIG['finetune_epochs']
        es_patience = CONFIG['finetune_es_patience']
        lr_patience = CONFIG['finetune_lr_patience']

    else:
        if finetune:
            raise FileNotFoundError(
                f'Fine-tune requested but no backbone found at: {CONFIG["backbone_path"]}\n'
                f'Copy cnn_backbone.h5 there before running with --finetune.'
            )
        backbone    = build_backbone(
            CONFIG['img_size'], CONFIG['channels'], CONFIG['embedding_dim']
        )
        lr          = CONFIG['learning_rate']
        epochs      = CONFIG['epochs']
        es_patience = 10
        lr_patience = 5

    siamese = build_siamese(backbone)
    # backbone.summary()
    siamese.compile(
        optimizer=Adam(learning_rate=lr),
        loss=contrastive_loss(margin=CONFIG['margin'])
    )

    print(f'\n  Mode        : {"fine-tune" if finetune else "scratch"}')
    print(f'  LR          : {lr}')
    print(f'  Max epochs  : {epochs}')
    print(f'  ES patience : {es_patience}')
    print(f'  LR patience : {lr_patience}')

    callbacks = [
        HardNegativeMiningCallback(backbone, train_gen, train_map, n_positives),
        BackboneCheckpoint(backbone, CONFIG['ckpt_dir']),
        ModelCheckpoint(CONFIG['siamese_ckpt'], monitor='val_loss',
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=es_patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=lr_patience, min_lr=1e-6, verbose=1),
        CSVLogger(CONFIG['log_path'], append=finetune),
    ]

    print('\nTraining...')
    history = siamese.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    backbone.save(CONFIG['backbone_path'])
    print(f'Final backbone saved: {CONFIG["backbone_path"]}')

    print('\nCalibrating threshold on validation set...')
    val_dists, val_labels = get_distances(backbone, val_gen)
    best_thresh, val_acc  = find_best_threshold(val_dists, val_labels)
    print(f'  Val accuracy  : {val_acc*100:.2f}%  threshold={best_thresh:.4f}')

    print('\nEvaluating on test set...')
    test_dists, test_labels = get_distances(backbone, test_gen)
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

    export_tflite(backbone, train_map, CONFIG['tflite_path'])

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

    return backbone, history, best_thresh, metrics


# ────────────────────────────────
#  CLI
# ────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     required=True,  help='Path to processed_dataset/')
    parser.add_argument('--finetune', action='store_true')
    args = parser.parse_args()

    backbone, history, threshold, metrics = train(
        processed_dir=args.data,
        finetune=args.finetune
    )

    print('\n' + '=' * 60)
    print('DONE')
    print(f'  TFLite    : {CONFIG["tflite_path"]}')
    print(f'  Config    : {CONFIG["deploy_config"]}')
    print(f'  Threshold : {threshold:.4f}')
    print(f'  EER       : {metrics["eer"]*100:.2f}%')
    print('=' * 60)