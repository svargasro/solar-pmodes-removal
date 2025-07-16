#!/usr/bin/env python3
# preprocess_optimized.py

import os
import glob
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.map import Map
from sunpy.physics.differential_rotation import differential_rotate
from sunpy.time import parse_time

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
def crop_and_save(path, observer, out_dir, crop_lim, is_first=False, index=0, total=1):
    fname = os.path.basename(path)
    progress = (index + 1) / total * 100
    print(f"[{index + 1}/{total}] ({progress:.1f}%) Aligning {fname}...")
    smap = Map(path)
    m_rot = differential_rotate(smap, observer=observer)

    print(f"[{index + 1}/{total}] ({progress:.1f}%) Cropping {fname}...")
    bl = SkyCoord(-crop_lim, -crop_lim, frame=m_rot.coordinate_frame)
    tr = SkyCoord(crop_lim, crop_lim, frame=m_rot.coordinate_frame)
    m_crop = m_rot.submap(bottom_left=bl, top_right=tr)
    m_crop.meta['DATE-OBS'] = parse_time(m_rot.date).isot #Agregado

    out_path = os.path.join(out_dir, f"dr_crop_{fname}")
    m_crop.save(out_path, overwrite=True)

    return m_crop if is_first else None

# Ejecutar procesamiento rápido
cropped_first = None
total_files = len(files)

for i, fpath in enumerate(files):
    cropped = crop_and_save(
        fpath,
        ref_observer,
        OUTPUT_DIR,
        CROP_LIM,
        is_first=(i == 0),
        index=i,
        total=total_files
    )
    if cropped is not None:
        cropped_first = cropped

# # -------- VISUALIZAR ORIGINAL Y PRIMER CROP --------
# plt.close('all')
# fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': first_map.wcs})

# axes[0].imshow(first_map.data, origin='lower', cmap='gray')
# axes[0].set_title("Original (full disk)")
# axes[0].set_xlabel('Helioprojective Lon')
# axes[0].set_ylabel('Helioprojective Lat')

# axes[1].imshow(cropped_first.data, origin='lower', cmap='gray')
# axes[1].set_title(f"Cropped ±{CROP_LIM.value}\" arsec")
# axes[1].set_xlabel('Helioprojective Lon')
# axes[1].set_ylabel('Helioprojective Lat')

# cbar = fig.colorbar(axes[1].images[0], ax=axes, orientation='horizontal', pad=0.1)
# cbar.set_label('Intensidad [DN]')

# plt.show()

print("Procesamiento optimizado completado. Mapas guardados en:", OUTPUT_DIR)
