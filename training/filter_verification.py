import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sunpy.map import Map

# -------- CONFIGURACIÓN --------
FITS_DIR = "./../images_intensity/data_hmi_Ic_45s_crop_dr/"
RAW_CUBE_NPY = "cube_raw.npy"
FILTERED_CUBE_NPY = "./../filtering_algorithm/filtered_cube.npy"
DT = 45.0  # segundos de cadencia

# -------- 1) Leer FITS y crear cubo raw --------
fits_files = sorted(glob.glob(os.path.join(FITS_DIR, "*.fits")))
if not fits_files:
    raise FileNotFoundError(f"No FITS found in {FITS_DIR}")

# Cargar datos de cada FITS en un array
cube_raw = np.stack([Map(f).data for f in fits_files])  # shape: (T, H, W)

np.save(RAW_CUBE_NPY, cube_raw)
print(f"Cubo raw guardado en '{RAW_CUBE_NPY}' con forma {cube_raw.shape}")

# -------- 2) Cargar cubo raw y filtrado --------
cube_raw = np.load(RAW_CUBE_NPY)

# print("Frames encontrados:", cube_raw.shape[0])
# print("Dimensiones espacial:", cube_raw.shape[1:], "(H, W)")
# print(f"Cubo raw guardado en '{RAW_CUBE_NPY}' con forma {cube_raw.shape}")

cube_filt = np.load(FILTERED_CUBE_NPY)
T, H, W = cube_raw.shape

# -------- 3) Selección de píxel para análisis --------
y0, x0 = H//2, W//2  # pijxel central

# -------- 4) Serie temporal y FFT --------
signal_raw = cube_raw[:, y0, x0]
signal_filt = cube_filt[:, y0, x0]

# Centrar señales
signal_raw -= signal_raw.mean()
signal_filt -= signal_filt.mean()

# FFT temporal
freq = np.fft.rfftfreq(T, d=DT)
spec_raw = np.abs(np.fft.rfft(signal_raw, norm='ortho'))
spec_filt = np.abs(np.fft.rfft(signal_filt, norm='ortho'))

# -------- 5) Graficar espectros --------
plt.figure(figsize=(8, 5))
plt.plot(freq * 1e3, spec_raw, label="Crudo")
plt.plot(freq * 1e3, spec_filt, label="Filtrado")
plt.axvline(3.3, color="r", linestyle="--", label="p-mode ~3.3 mHz")
plt.xlim(0, 7)
plt.xlabel("Frecuencia [mHz]")
plt.ylabel("Amplitud FFT")
plt.title("Espectro temporal en píxel central")
plt.legend()
plt.grid(True)
plt.show()

# -------- 6) Cálculo de potencia en banda p-modes --------
band = (freq >= 2e-3) & (freq <= 5e-3)
power_raw = np.sum(spec_raw[band]**2)
power_filt = np.sum(spec_filt[band]**2)
suppression = 100 * (1 - power_filt / power_raw)

print(f"Potencia p-modes (2–5 mHz) crudo:   {power_raw:.2e}")
print(f"Potencia p-modes (2–5 mHz) filtrado: {power_filt:.2e}")
print(f"Supresión de p-modes: {suppression:.1f}%")
