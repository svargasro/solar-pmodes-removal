#!/usr/bin/env python3
# preprocess.py

import os
import glob
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u

from sunpy.map import Map
from sunpy.physics.differential_rotation import differential_rotate

# -------- PARÁMETROS --------
input_dir  = "data_hmi_Ic_45s/"
output_dir = "data_hmi_Ic_45s_crop_dr/"
crop_lim   = 500 * u.arcsec   # ±500 arcsec en X e Y

# -------- CREA LAS CARPETAS --------
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# -------- LISTA DE ARCHIVOS --------
files = sorted(glob.glob(os.path.join(input_dir, "*.fits")))
if not files:
    raise FileNotFoundError(f"No se encontró ningún FITS en {input_dir}")

# -------- 1) Leer mapas --------
maps = [Map(f) for f in files]

# -------- 2) Determinar tiempo de referencia --------
# Usamos la fecha del primer mapa como 'time' objetivo
ref_time = maps[0].date

# -------- 3) Rotación diferencial de cada mapa al tiempo ref_time --------
aligned = []
for m in maps:
    # Solo especificamos time; se usa el observer del mapa original
    m_rot = differential_rotate(m, time=ref_time)
    aligned.append(m_rot)

# -------- 4) Función de recorte helioproyectivo --------
def crop_map(smap, lim):
    bl = SkyCoord(-lim, -lim, frame=smap.coordinate_frame)
    tr = SkyCoord( lim,  lim, frame=smap.coordinate_frame)
    return smap.submap(bl, tr)

# -------- 5) Recortar y guardar mapas --------
for m in aligned:
    m_crop = crop_map(m, crop_lim)
    fname = os.path.basename(m.filenames[0])
    out_path = os.path.join(output_dir, "dr_crop_" + fname)
    m_crop.save(out_path, overwrite=True)

# -------- 6) Visualización antes/después del primer frame --------
orig = maps[0]
crpd = crop_map(aligned[0], crop_lim)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
orig.plot(axes=ax1, cmap="gray", title="Original (full disk)")
ax1.set_axis_off()

crpd.plot(axes=ax2, cmap="gray", title=f"Cropped ±{crop_lim.value}\"")
ax2.set_axis_off()

plt.tight_layout()
plt.show()

print("Procesamiento completado. Mapas guardados en:", output_dir)
