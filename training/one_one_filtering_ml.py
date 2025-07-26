#!/usr/bin/env python3

#!/usr/bin/env python3
import os, glob, gc
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, metrics
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import BatchNormalization, Dropout

# -------- Parámetros --------
INPUT_RAW_NPY   = "cube_raw.npy"
INPUT_FILT_NPY  = "filtered_cube.npy"
OUTPUT_MODEL    = "autoencoder2d.keras"
USE_ORIGINAL    = False          # True = usar resolución nativa; False = down‑sample
TARGET_H, TARGET_W = 512, 512

# -------- Mixed precision (GPU) --------
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# -------- Leer metadata para formas --------
raw_shape = np.load(INPUT_RAW_NPY, mmap_mode='r').shape  # (T, H, W)
T, H_full, W_full = raw_shape

# Determinar forma de entrada al modelo
if USE_ORIGINAL:
    H_model, W_model = H_full, W_full
else:
    H_model, W_model = TARGET_H, TARGET_W

# -------- Función de carga y pre‑procesado --------
def load_pair(i):
    """
    Carga la i-ésima imagen raw y filtrada de sus .npy,
    opcionalmente redimensiona y normaliza.
    """
    raw  = np.load(INPUT_RAW_NPY, mmap_mode='r')[i]
    filt = np.load(INPUT_FILT_NPY, mmap_mode='r')[i]

    # añadir canal
    raw  = raw[..., None]
    filt = filt[..., None]

    if not USE_ORIGINAL:
        raw  = tf.image.resize(raw,  [TARGET_H, TARGET_W],
                               method='bilinear').numpy()
        filt = tf.image.resize(filt, [TARGET_H, TARGET_W],
                               method='bilinear').numpy()

    # normalizar ambas con el mismo max
    maxv = raw.max()
    raw  = (raw  / maxv).astype(np.float32)
    filt = (filt / maxv).astype(np.float32)

    return raw, filt

def tf_load_pair(i):
    """
    Envoltorio para tf.data: llama a load_pair y
    fija la forma de los tensores resultantes.
    """
    raw, filt = tf.py_function(
        func=load_pair,
        inp=[i],
        Tout=[tf.float32, tf.float32]
    )
    # De forma explícita decimos a TF la forma esperada:
    raw.set_shape((H_model, W_model, 1))
    filt.set_shape((H_model, W_model, 1))
    return raw, filt

# -------- Construir tf.data.Dataset --------
indices = tf.data.Dataset.range(T)
ds = (indices
      .map(tf_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
      .batch(1, drop_remainder=True)
      .prefetch(tf.data.AUTOTUNE)
)

# liberar referencias
gc.collect()

# -------- Definir un autoencoder 2D ligero --------
def build_autoencoder2d(H, W, C=1):
    inp = layers.Input((H, W, C))
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(inp)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2, name='bottleneck')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(x)
    out = layers.Conv2D(C, 3, padding='same', activation='linear')(x)
    return models.Model(inp, out, name='Autoencoder2D')

model = build_autoencoder2d(H_model, W_model, 1)
model.summary()

# -------- Compilación --------
opt = optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=opt,
    loss='mse',
    metrics=[metrics.MeanSquaredError(name='mse')]
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# -------- Entrenamiento --------
model.fit(
    ds,
    epochs=25,
    callbacks=[reduce_lr]
)

# -------- Guardar modelo --------
model.save(OUTPUT_MODEL)
print(f"Modelo guardado en {OUTPUT_MODEL}")
