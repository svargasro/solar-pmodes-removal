#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Cropping3D

# -------- Parámetros físicos --------
scale  = 1814.544    # arcsec/pixel
t_step = 45.0        # segundos entre frames
v_ph   = 4.0         # km/s umbral de fase

# -------- Cargar cubos --------
cube_raw  = np.load("cube_raw.npy")      # shape (T_raw, H, W)
cube_filt = np.load("filtered_cube.npy") # shape (T_filt, H, W)

T_raw, H, W = cube_raw.shape
T_filt = cube_filt.shape[0]

# Ajustar frames si es impar
if T_raw % 2 != 0:
    print(f"{T_raw} frames detectados (impar). Descartando el último.")
    cube_raw = cube_raw[:-1]

# Validar forma
assert cube_raw.shape[0] == cube_filt.shape[0]
T = cube_raw.shape[0]

# -------- Redimensionar para ahorrar memoria --------
TARGET_H, TARGET_W = 512, 512  # <--- ajusta según recursos disponibles

def resize_cube(cube, new_h, new_w):
    return np.stack([tf.image.resize(c[..., np.newaxis], [new_h, new_w], method='bilinear')[..., 0].numpy() for c in cube])

cube_raw_small  = resize_cube(cube_raw, TARGET_H, TARGET_W)
cube_filt_small = resize_cube(cube_filt, TARGET_H, TARGET_W)

x_train = cube_raw_small[np.newaxis, ..., np.newaxis].astype(np.float16)
y_train = cube_filt_small[np.newaxis, ..., np.newaxis].astype(np.float16)

# -------- Autoencoder --------
def build_small_autoencoder(T, H, W, C):
    inp = tf.keras.Input(shape=(T, H, W, C))
    x = layers.Conv3D(8, (3,3,3), padding='same', activation='relu')(inp)
    x = layers.MaxPool3D((1,2,2))(x)
    x = layers.Conv3D(16, (3,3,3), padding='same', activation='relu')(x)
    x = layers.MaxPool3D((2,2,2))(x)
    x = layers.Conv3D(32, (3,3,3), padding='same', activation='relu')(x)
    encoded = layers.MaxPool3D((2,2,2), name='bottleneck')(x)
    x = layers.Conv3DTranspose(32, (3,3,3), strides=(2,2,2), padding='same', activation='relu')(encoded)
    x = layers.Conv3DTranspose(16, (3,3,3), strides=(2,2,2), padding='same', activation='relu')(x)
    x = layers.Conv3DTranspose(8, (3,3,3), strides=(1,2,2), padding='same', activation='relu')(x)
    decoded = layers.Conv3D(C, (3,3,3), activation='linear', padding='same')(x)
    return models.Model(inp, decoded)

model = build_small_autoencoder(T, TARGET_H, TARGET_W, 1)
model.summary()

# -------- Máscara de pérdida física --------
def phase_velocity_mask_np(shape, scale, t_step, v_ph=4.0):
    T, H, W = shape
    kx = np.fft.fftshift(np.fft.fftfreq(W, d=scale * 725.0))
    ky = np.fft.fftshift(np.fft.fftfreq(H, d=scale * 725.0))
    w  = np.fft.fftshift(np.fft.fftfreq(T, d=t_step))
    KX, KY, Wf = np.meshgrid(kx, ky, w, indexing='xy')
    k_mag = np.sqrt(KX**2 + KY**2)
    mask = (np.abs(Wf) > k_mag * v_ph).astype(np.float32)
    return tf.convert_to_tensor(mask, dtype=tf.float32)

mask = phase_velocity_mask_np((T, TARGET_H, TARGET_W), scale, t_step, v_ph)
mask = tf.reshape(mask, (1, T, TARGET_H, TARGET_W, 1))

@tf.function
def physics_loss(y_true, y_pred):
    fft_pred = tf.signal.fft3d(tf.cast(y_pred, tf.complex64))
    return tf.reduce_mean(tf.abs(fft_pred)**2 * mask)

# -------- Compilar y entrenar --------
lambda_phys = 1e-3
model.compile(
    optimizer='adam',
    loss=lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred)) + lambda_phys * physics_loss(y_true, y_pred)
)

model.fit(x_train, y_train, epochs=30, batch_size=1, shuffle=False)

# -------- Inferencia --------
filt_output = model.predict(x_train)
piml_cube = np.squeeze(filt_output, axis=(0, -1))  # (T, H, W)
np.save('cube_piml_filtered_small.npy', piml_cube)
print('Cubo PIML guardado como cube_piml_filtered_small.npy')
