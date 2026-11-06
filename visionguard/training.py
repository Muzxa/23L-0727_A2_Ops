"""tf.data, model definitions, reproducible fitting, and saved training evidence."""
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def environment(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                  tensorflow=tf.__version__, keras=keras.__version__,
                  gpu=[x.name for x in tf.config.list_physical_devices('GPU')],
                  build=tf.sysconfig.get_build_info(), platform=sys.platform)
    (out / 'environment.json').write_text(json.dumps(record, indent=2, default=str))
    (out / 'requirements-lock.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    print(json.dumps(record, indent=2, default=str))
    return record


def dataset(split_csv, root, size=224, batch=32, training=False, seed=42):
    frame = pd.read_csv(split_csv)
    paths = [str(Path(root) / p) for p in frame.path]
    ds = tf.data.Dataset.from_tensor_slices((paths, frame.label.to_numpy(np.float32)))
    def decode(path, label):
        im = tf.io.decode_png(tf.io.read_file(path), channels=3)
        # Letterboxing preserves aspect ratio. Padding is zero; no geometric distortion.
        im = tf.image.resize_with_pad(im, size, size, method='bilinear', antialias=True)
        return im, tf.reshape(label, [1])
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    # Cache deterministic resized pixels, then reshuffle each training epoch.
    # The assignment's 680 training images occupy about 410 MB at 224px.
    # Augmentation stays inside the model and is never cached.
    ds = ds.map(decode, num_parallel_calls=2).with_options(opts).cache()
    if training:
        ds = ds.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(1)


def augmentation(seed=42):
    return keras.Sequential([
        keras.layers.RandomFlip('horizontal', seed=seed),
        keras.layers.RandomRotation(.025, fill_mode='reflect', seed=seed+1),
        keras.layers.RandomZoom(.08, fill_mode='reflect', seed=seed+2),
        keras.layers.RandomContrast(.08, seed=seed+3),
    ], name='train_only_augmentation')


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='binary_crossentropy',
                  metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                           keras.metrics.AUC(name='auc'),
                           keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')])
    return model


def baseline(size=224, seed=42):
    inputs = keras.Input((size, size, 3), name='rgb_pixels_0_255')
    x = augmentation(seed)(inputs)
    x = keras.layers.Rescaling(1/255., name='scale_0_1')(x)
    for i, filters in enumerate([32, 64, 128], 1):
        x = keras.layers.Conv2D(filters, 3, padding='same', use_bias=False, name=f'conv_{i}')(x)
        x = keras.layers.BatchNormalization(name=f'bn_{i}')(x)
        x = keras.layers.Activation('relu', name=f'relu_{i}')(x)
        x = keras.layers.MaxPooling2D(name=f'pool_{i}')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(.4)(x)
    output = keras.layers.Dense(1, activation='sigmoid', name='p_ai')(x)
    return compile_model(keras.Model(inputs, output, name='scratch_cnn'), 1e-3)


def transfer(size=224, dropout=.3, seed=42, weights='imagenet'):
    inputs = keras.Input((size, size, 3), name='rgb_pixels_0_255')
    x = augmentation(seed)(inputs)
    # Exactly equivalent to MobileNetV2 preprocess_input: (pixels / 127.5) - 1.
    x = keras.layers.Rescaling(1/127.5, offset=-1, name='mobilenet_preprocess')(x)
    backbone = keras.applications.MobileNetV2(include_top=False, weights=weights, input_shape=(size, size, 3))
    backbone.trainable = False
    # Keep BN moving statistics fixed during both stages.
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout, name='head_dropout')(x)
    output = keras.layers.Dense(1, activation='sigmoid', name='p_ai')(x)
    return compile_model(keras.Model(inputs, output, name='mobilenetv2_classifier'), 1e-3)


def unfreeze_last_block(model, count=None):
    backbone = model.get_layer('mobilenetv2_1.00_224') if any(
        l.name == 'mobilenetv2_1.00_224' for l in model.layers) else next(
        l for l in model.layers if isinstance(l, keras.Model) and 'mobilenetv2' in l.name)
    backbone.trainable = True
    eligible = [l for l in backbone.layers if
                (l.name.startswith('block_16_') or l.name == 'Conv_1')
                and l.weights and not isinstance(l, keras.layers.BatchNormalization)]
    chosen = eligible if count is None else eligible[-count:]
    names = {l.name for l in chosen}
    for layer in backbone.layers:
        layer.trainable = layer.name in names
    return [l.name for l in chosen]


def layer_evidence(model, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    model.summary(print_fn=lambda s, **kw: lines.append(s), expand_nested=True)
    (out / 'model_summary.txt').write_text('\n'.join(lines))
    rows, trainable = [], []
    def walk(layers, prefix=''):
        for l in layers:
            name = prefix + l.name
            rows.append(dict(layer=name, type=type(l).__name__,
                             output_shape=str(getattr(l, 'output', '') .shape if hasattr(getattr(l, 'output', None), 'shape') else ''),
                             parameters=l.count_params(), trainable=l.trainable))
            if isinstance(l, keras.Model):
                walk(l.layers, name + '/')
            elif l.trainable and l.trainable_weights:
                trainable.append(name)
    walk(model.layers)
    pd.DataFrame(rows).to_csv(out / 'layers.csv', index=False)
    (out / 'trainable_layers.json').write_text(json.dumps(trainable, indent=2))
    print('\n'.join(lines))


def fit_stage(model, train, val, out, epochs=15):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    layer_evidence(model, out)
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
                 keras.callbacks.ModelCheckpoint(str(out / 'best.keras'), monitor='val_loss', save_best_only=True),
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=2, min_lr=1e-7),
                 keras.callbacks.CSVLogger(str(out / 'epochs.csv'))]
    h = model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks, verbose=2).history
    best = int(np.argmin(h['val_loss'])) + 1
    (out / 'history.json').write_text(json.dumps(h, indent=2))
    info = dict(best_epoch=best, best_val_loss=float(min(h['val_loss'])), epochs_run=len(h['loss']))
    (out / 'best_epoch.json').write_text(json.dumps(info, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        ax.plot(range(1, len(h[metric])+1), h[metric], marker='o', label='train')
        ax.plot(range(1, len(h[metric])+1), h['val_'+metric], marker='o', label='validation')
        ax.axvline(best, linestyle='--', color='gray', label=f'Best val-loss epoch: {best}')
        ax.set(xlabel='Epoch', ylabel=metric, title=out.name)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'curves.png', dpi=160)
    plt.close(fig)
    return keras.models.load_model(out / 'best.keras'), info


def augmented_batch(ds, out, seed=42):
    images, labels = next(iter(ds))
    aug = augmentation(seed)(images, training=True).numpy()
    n = min(12, len(aug))
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i < n:
            ax.imshow(np.clip(aug[i], 0, 255).astype('uint8'))
            ax.set_title('AI' if labels[i].numpy().item() else 'Real')
    fig.suptitle('Training augmentation preview (not used for validation/test)')
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
