#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# -------- CONFIGURACIÓN --------
RAW_CUBE_NPY            = "cube_raw.npy"
FILTERED_CUBE_NPY_ML    = "ml_cube.npy"
FILTERED_CUBE_NPY       = "filtered_cube.npy"
DT                      = 45.0  # segundos de cadencia

# -------- 2) Cargar cubos --------
cube_raw   = np.load(RAW_CUBE_NPY)       # (T, H, W)
cube_ml    = np.load(FILTERED_CUBE_NPY_ML)  # (T, H, W)
cube_ref   = np.load(FILTERED_CUBE_NPY)     # (T, H, W)

T, H, W = cube_raw.shape

# -------- 3) Selección de píxel central --------
y0, x0 = H//2, W//2

# -------- 4) Serie temporal y FFT --------
signal_raw  = cube_raw[:, y0, x0]
signal_ml   = cube_ml[:,  y0, x0]
signal_ref  = cube_ref[:, y0, x0]

# centrar
signal_raw -= signal_raw.mean()
signal_ml  -= signal_ml.mean()
signal_ref -= signal_ref.mean()

# FFT temporal
freq      = np.fft.rfftfreq(T, d=DT)
spec_raw  = np.abs(np.fft.rfft(signal_raw, norm='ortho'))
spec_ml   = np.abs(np.fft.rfft(signal_ml,  norm='ortho'))
spec_ref  = np.abs(np.fft.rfft(signal_ref, norm='ortho'))

# -------- 5) Graficar espectros --------
plt.figure(figsize=(8,5))
plt.plot(freq*1e3, spec_raw, label="Crudo")
plt.plot(freq*1e3, spec_ref, label="Filtrado (referencia)")
plt.plot(freq*1e3, spec_ml,  label="Filtrado (ML)")
plt.axvline(3.3, color='r', linestyle='--', label="p‑mode ~3.3 mHz")
plt.xlim(0,7)
plt.xlabel("Frecuencia [mHz]")
plt.ylabel("Amplitud FFT")
plt.title("Espectro temporal en píxel central")
plt.legend()
plt.grid(True)
plt.show()

# -------- 6) Cálculo de potencia en banda p-modes --------
band = (freq >= 2e-3) & (freq <= 5e-3)
power_raw = np.sum(spec_raw[band]**2)
power_ref = np.sum(spec_ref[band]**2)
power_ml  = np.sum(spec_ml[band]**2)

supp_ref = 100*(1 - power_ref/power_raw)
supp_ml  = 100*(1 - power_ml /power_raw)

print(f"Potencia p-modes (2–5 mHz) crudo:        {power_raw:.2e}")
print(f"Potencia p-modes (2–5 mHz) referencia:  {power_ref:.2e}  supresión {supp_ref:.1f}%")
print(f"Potencia p-modes (2–5 mHz) ML:          {power_ml:.2e}  supresión {supp_ml:.1f}%")
