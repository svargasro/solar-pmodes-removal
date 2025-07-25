#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

# -------- CONFIGURACIÓN --------
MODEL_PATH = "autoencoder2d.keras"
RAW_CUBE_PATH = "cube_raw.npy"
OUTPUT_CUBE_PATH = "ml_cube.npy"
USE_ORIGINAL_RESOLUTION = False  # ← CAMBIA ESTO a True si quieres usar resolución original
TARGET_H, TARGET_W = 512, 512    # tamaño usado si no se usa resolución original

# -------- CARGAR MODELO Y CUBO --------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
cube_raw = np.load(RAW_CUBE_PATH)
T, H_orig, W_orig = cube_raw.shape

# -------- NORMALIZAR --------
maxv = np.max(cube_raw)
cube_raw = cube_raw / maxv

# -------- PROCESAMIENTO E INFERENCIA --------
ml_cube = []

for i in range(T):
    frame = cube_raw[i]  # (H, W)

    if USE_ORIGINAL_RESOLUTION:
        resized = frame  # sin cambios
        input_tensor = frame[None, ..., None].astype(np.float32)
    else:
        resized = tf.image.resize(frame[..., None], [TARGET_H, TARGET_W],
                                  method='bilinear')[..., 0].numpy()
        input_tensor = resized[None, ..., None].astype(np.float32)

    # Inferencia
    pred = model.predict(input_tensor)[0, ..., 0]
    pred *= maxv  # volver a escala original

    if USE_ORIGINAL_RESOLUTION:
        restored = pred  # sin cambios
    else:
        restored = tf.image.resize(pred[..., None], [H_orig, W_orig],
                                   method='bilinear')[..., 0].numpy()

    ml_cube.append(restored)

# -------- GUARDAR --------
ml_cube = np.stack(ml_cube)  # (T, H_orig, W_orig)
np.save(OUTPUT_CUBE_PATH, ml_cube)
print(f"Cubo generado guardado en '{OUTPUT_CUBE_PATH}'")
