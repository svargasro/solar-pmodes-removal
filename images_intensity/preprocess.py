#!/usr/bin/env python3
# preprocess_from_folder.py

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from sunpy.map import Map
from sunpy.image.coalignment import mapsequence_coalignment
from sunpy.time import parse_time

# -------- PARÁMETROS --------
input_dir  = "data_hmi_Ic_45s/"
output_dir = "data_hmi_Ic_45s_proc/"
poly_deg   = 3       # grado del polinomio para limb-darkening
n_bins     = 100     # número de anillos para perfil radial

# -------- CREA LAS CARPETAS --------
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# -------- LISTA DE ARCHIVOS --------
files = sorted(glob.glob(os.path.join(input_dir, "*.fits")))
if not files:
    raise FileNotFoundError(f"No se encontró ningún FITS en {input_dir}")

# -------- 1) Leer como MapSequence --------
print(f"Cargando {len(files)} mapas desde {input_dir}")
maps = [Map(f) for f in files]

# -------- 2) Co‑alineación al primer frame --------
print("Co‑alineando imágenes…")
aligned = mapsequence_coalignment.mapsequence_coalignment(maps, reference=maps[0])

# -------- 3) Función de corrección de limb‑darkening --------
def correct_limb_darkening(smap, poly_deg=3, n_bins=100):
    data = smap.data.astype(float)
    y, x = np.indices(data.shape)
    cx, cy = smap.reference_pixel.x.value, smap.reference_pixel.y.value
    r = np.sqrt((x-cx)**2 + (y-cy)**2) / smap.rsun_pix

    # Construir perfil radial
    bins = np.linspace(0, 1, n_bins+1)
    centers = 0.5*(bins[:-1] + bins[1:])
    medians = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i+1])
        medians[i] = np.median(data[mask]) if np.any(mask) else np.nan

    ok = ~np.isnan(medians)
    coeff = np.polyfit(centers[ok], medians[ok], poly_deg)
    limb = np.polyval(coeff, r)

    corrected = np.zeros_like(data)
    mask_disk = r <= 1
    corrected[mask_disk] = data[mask_disk] / limb[mask_disk]

    return smap._new_instance(corrected, smap.meta)

# -------- 4) Aplicar corrección y guardar --------
processed = []
print("Aplicando limb‑darkening y guardando…")
for m, f in zip(aligned, files):
    m_corr = correct_limb_darkening(m, poly_deg=poly_deg, n_bins=n_bins)
    processed.append(m_corr)
    out_path = os.path.join(output_dir, "proc_" + os.path.basename(f))
    m_corr.save(out_path, overwrite=True)

# -------- 5) Visualización antes/después (opcional) --------
fig, axes = plt.subplots(1,2, figsize=(12,6))
axes[0].imshow(maps[0].data, cmap="gray", origin="lower")
axes[0].set_title("Original")
axes[0].axis("off")

im = axes[1].imshow(processed[0].data, cmap="gray",
                    vmin=np.percentile(processed[0].data,5),
                    vmax=np.percentile(processed[0].data,95),
                    origin="lower")
axes[1].set_title("Limb‑darkening corregido")
axes[1].axis("off")

plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
plt.tight_layout()
plt.show()

print("Procesamiento completado. Mapas guardados en:", output_dir)
