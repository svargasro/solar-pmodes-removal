#!/usr/bin/env python3
# visualize_crop_vs_original.py

import os
import glob
import matplotlib.pyplot as plt
from sunpy.map import Map

# -------- PARÁMETROS --------
INPUT_DIR  = "data_hmi_Ic_45s/"
CROP_DIR   = "data_hmi_Ic_45s_crop_dr/"
INDEX      = 0  # Cambia este índice para visualizar otra imagen

# -------- OBTENER ARCHIVOS --------
original_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.fits")))
crop_files     = sorted(glob.glob(os.path.join(CROP_DIR, "*.fits")))

if not original_files:
    raise FileNotFoundError(f"No FITS files in {INPUT_DIR}")
if not crop_files:
    raise FileNotFoundError(f"No FITS files in {CROP_DIR}")

# -------- ASEGURAR ÍNDICE VÁLIDO --------
if INDEX >= len(original_files):
    raise IndexError(f"Índice {INDEX} excede el número de archivos disponibles ({len(original_files)})")

# -------- CARGAR MAPAS --------
original_map = Map(original_files[INDEX])
cropped_map  = Map(crop_files[INDEX])

# -------- VISUALIZACIÓN --------
plt.close('all')
fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': original_map.wcs})

original_map.plot(axes=axes[0], cmap='gray')
axes[0].set_title("Original (full disk)")
axes[0].set_xlabel('Helioprojective Lon')
axes[0].set_ylabel('Helioprojective Lat')

cropped_map.plot(axes=axes[1], cmap='gray',autoalign=True)
axes[1].set_title("Recortado (±500)")
axes[1].set_xlabel('Helioprojective Lon')
axes[1].set_ylabel('Helioprojective Lat')

# cbar = fig.colorbar(axes[1].images[0], ax=axes, orientation='horizontal', pad=0.1)
# cbar.set_label('Intensidad [DN]')


plt.show()
