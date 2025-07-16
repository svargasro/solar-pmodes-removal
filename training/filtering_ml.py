#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# -------- Parámetros físicos --------
scale  = 1814.544    # arcsec/pixel
t_step = 45.0        # segundos entre frames
v_ph   = 4.0         # km/s umbral de fase

# -------- Cargar cubos desde .npy --------
cube_raw  = np.load("cube_raw.npy")      # shape (T_raw, H, W)
cube_filt = np.load("filtered_cube.npy") # shape (T_filt, H, W)

# -------- Igualar número de frames --------
T_raw, H, W = cube_raw.shape
T_filt       = cube_filt.shape[0]

if T_raw % 2 != 0:
    # Si hay un número impar de frames, descartamos el último del raw
    print(f"{T_raw} frames detectados (impar). Descartando el último para emparejar.")
    cube_raw = cube_raw[:-1]
    T_raw -= 1

# Ahora comprobamos que coincidan en T
if cube_raw.shape[0] != cube_filt.shape[0]:
    raise ValueError(f"Tras el ajuste raw tiene {cube_raw.shape[0]} frames "
                     f"pero filtrado tiene {cube_filt.shape[0]}.")

T = cube_raw.shape[0]
print(f"Usando {T} frames para entrenamiento (H={H}, W={W}).")

# Añadir batch y canal: (N=1, T, H, W, C=1)
x_train = cube_raw[np.newaxis, ..., np.newaxis].astype(np.float32)
y_train = cube_filt[np.newaxis, ..., np.newaxis].astype(np.float32)

# -------- Construcción del autoencoder 3D --------
def build_subsonic_autoencoder(T, H, W, C):
    inp = tf.keras.Input(shape=(T, H, W, C), name='input_cube')
    x = layers.Conv3D(16, (3,3,3), padding='same', activation='relu')(inp)
    x = layers.MaxPool3D((1,2,2), padding='same')(x)
    x = layers.Conv3D(32, (3,3,3), padding='same', activation='relu')(x)
    x = layers.MaxPool3D((2,2,2), padding='same')(x)
    x = layers.Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
    encoded = layers.MaxPool3D((2,2,2), padding='same', name='bottleneck')(x)
    x = layers.Conv3DTranspose(64, (3,3,3), strides=(2,2,2), padding='same', activation='relu')(encoded)
    x = layers.Conv3DTranspose(32, (3,3,3), strides=(2,2,2), padding='same', activation='relu')(x)
    x = layers.Conv3DTranspose(16, (3,3,3), strides=(1,2,2), padding='same', activation='relu')(x)
    decoded = layers.Conv3D(C, (3,3,3), activation='linear', padding='same', name='output_cube')(x)
    return models.Model(inp, decoded, name='SubsonicAutoencoder')

model = build_subsonic_autoencoder(time_steps, height, width, channels)
model.summary()

# -------- Función de pérdida física --------
def phase_velocity_mask_tf(shape, scale, t_step, v_ph=4.0):
    T, H, W = shape
    kx = tf.signal.fftshift(tf.signal.fftfreq(W, d=scale*725.0))
    ky = tf.signal.fftshift(tf.signal.fftfreq(H, d=scale*725.0))
    w  = tf.signal.fftshift(tf.signal.rfftfreq(T, d=t_step))
    KX, KY, Wf = tf.meshgrid(kx, ky, w, indexing='xy')
    k_mag = tf.sqrt(KX**2 + KY**2)
    return tf.cast(Wf > k_mag * v_ph, tf.float32)

@tf.function
def physics_loss(y_true, y_pred):
    fft_pred = tf.signal.rfftn(y_pred, axes=(1,2,3), norm='ortho')
    mask = phase_velocity_mask_tf((time_steps, height, width), scale, t_step)
    mask = tf.reshape(mask, (1,) + mask.shape + (1,))
    return tf.reduce_mean(tf.abs(fft_pred)**2 * mask)

# -------- Compilar y entrenar --------
lambda_phys = 1e-3
model.compile(
    optimizer='adam',
    loss=lambda y_true, y_pred:
         tf.reduce_mean(tf.square(y_true - y_pred))
         + lambda_phys * physics_loss(y_true, y_pred)
)

# Entrenamiento
epochs = 50
batch_size = 1
model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          shuffle=False)

# -------- Inferencia --------
filt_output = model.predict(x_train)
piml_cube = np.squeeze(filt_output, axis=(0,-1))  # (T, H, W)
np.save('cube_piml_filtered.npy', piml_cube)
print('Cubo PIML guardado en cube_piml_filtered.npy')
