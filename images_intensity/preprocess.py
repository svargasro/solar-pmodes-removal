#!/usr/bin/env python3
import os
import glob
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.map import Map
from sunpy.physics.differential_rotation import differential_rotate
from sunpy.time import parse_time

# -------- CONFIGURACIÓN --------
INPUT_DIR  = "./data_hmi_Ic_45s/"
OUTPUT_DIR = "./data_hmi_Ic_45s_crop_dr/"
CROP_LIM   = 500 * u.arcsec   # ±500 arcsec en X e Y

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Listado de archivos y referencia
files        = sorted(glob.glob(os.path.join(INPUT_DIR, "*.fits")))
first_map    = Map(files[0])
ref_observer = first_map.observer_coordinate

# Muestra para fijar dimensiones de píxeles
bl_s = SkyCoord(-CROP_LIM, -CROP_LIM, frame=first_map.coordinate_frame)
tr_s = SkyCoord( CROP_LIM,  CROP_LIM, frame=first_map.coordinate_frame)
sample = first_map.submap(bottom_left=bl_s, top_right=tr_s)
ny_ref, nx_ref = sample.data.shape
print(f"→ Tamaño fijo de recorte (pixeles): {ny_ref}×{nx_ref}")

def crop_and_save(path, observer, out_dir, crop_lim, idx, total):
    fname = os.path.basename(path)
    prog  = (idx+1)/total*100
    print(f"[{idx+1}/{total}] {prog:.1f}% – {fname}")

    # 1) Leer y rotar diferencialmente
    smap  = Map(path)
    m_rot = differential_rotate(smap, observer=observer)

    # 2) Definir esquinas en coordenadas
    bl = SkyCoord(-crop_lim, -crop_lim, frame=m_rot.coordinate_frame)
    tr = SkyCoord( crop_lim,  crop_lim, frame=m_rot.coordinate_frame)

    # 3) Hacer submap de la ventana helioprojectiva
    m_sub = m_rot.submap(bottom_left=bl, top_right=tr)

    # 4) Remuestrear a la forma fija en píxeles
    m_crop = m_sub.resample((ny_ref, nx_ref) * u.pix)

    # 5) Mantener la fecha original
    m_crop.meta['DATE-OBS'] = parse_time(m_rot.date).isot

    # 6) Guardar
    out = os.path.join(out_dir, f"dr_crop_{fname}")
    m_crop.save(out, overwrite=True)

# Ejecutar para todos los archivos
total = len(files)
for i, f in enumerate(files):
    crop_and_save(f, ref_observer, OUTPUT_DIR, CROP_LIM, i, total)

print("Preprocesamiento finalizado")
