#!/usr/bin/env python3
# preprocess_optimized.py

import os
import glob
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.map import Map
from sunpy.physics.differential_rotation import differential_rotate

# -------- PARÁMETROS --------
INPUT_DIR  = "data_hmi_Ic_45s/"
OUTPUT_DIR = "data_hmi_Ic_45s_crop_dr/"
CROP_LIM   = 500 * u.arcsec   # ±500 arcsec en X e Y

# -------- PREPARAR DIRECTORIOS --------
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- LISTAR FITS --------
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.fits")))
if not files:
    raise FileNotFoundError(f"No FITS found in {INPUT_DIR}")

# -------- LEER PRIMER MAPA PARA REFERENCIA --------
first_map = Map(files[0])
ref_observer = first_map.observer_coordinate

# -------- PROCESAR CADA ARCHIVO INDIVIDUALMENTE --------
# Solo mantenemos en memoria un mapa a la vez
def crop_and_save(path, observer, out_dir, crop_lim, is_first=False):
    smap = Map(path)
    # Rotación diferencial al observador de referencia
    m_rot = differential_rotate(smap, observer=observer)
    # Recorte
    bl = SkyCoord(-crop_lim, -crop_lim, frame=m_rot.coordinate_frame)
    tr = SkyCoord(crop_lim, crop_lim, frame=m_rot.coordinate_frame)
    m_crop = m_rot.submap(bottom_left=bl, top_right=tr)
    # Guardar
    fname = os.path.basename(path)
    m_crop.save(os.path.join(out_dir, f"dr_crop_{fname}"), overwrite=True)
    # Devolver el primer recortado para plot
    return m_crop if is_first else None

# Ejecutar procesamiento rápido
cropped_first = None
for i, fpath in enumerate(files):
    cropped = crop_and_save(
        fpath,
        ref_observer,
        OUTPUT_DIR,
        CROP_LIM,
        is_first=(i==0)
    )
    if cropped is not None:
        cropped_first = cropped

# -------- VISUALIZAR ORIGINAL Y PRIMER CROP --------
plt.close('all')
fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': first_map.wcs})

# Original full disk
axes[0].imshow(first_map.data, origin='lower', cmap='gray')
axes[0].set_title("Original (full disk)")
axes[0].set_xlabel('Helioprojective Lon')
axes[0].set_ylabel('Helioprojective Lat')

# Primer recortado
axes[1].imshow(cropped_first.data, origin='lower', cmap='gray')
axes[1].set_title(f"Cropped ±{CROP_LIM.value}\" arsec")
axes[1].set_xlabel('Helioprojective Lon')
axes[1].set_ylabel('Helioprojective Lat')

# Colorbar horizontal para ambas
cbar = fig.colorbar(axes[1].images[0], ax=axes, orientation='horizontal', pad=0.1)
cbar.set_label('Intensidad [DN]')

plt.show()

print("Procesamiento optimizado completado. Mapas guardados en:", OUTPUT_DIR)
