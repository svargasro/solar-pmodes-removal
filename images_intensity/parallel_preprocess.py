#!/usr/bin/env python3
import os
import glob
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.map import Map
from sunpy.physics.differential_rotation import differential_rotate
from sunpy.time import parse_time
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------- CONFIGURACIÓN --------
INPUT_DIR  = "data_hmi_Ic_45s/"
OUTPUT_DIR = "data_hmi_Ic_45s_crop_dr/"
CROP_LIM   = 500 * u.arcsec   # ±500 arcsec en X e Y

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Listado de archivos y referencia
fits_files   = sorted(glob.glob(os.path.join(INPUT_DIR, "*.fits")))
if not fits_files:
    raise FileNotFoundError(f"No FITS found in {INPUT_DIR}")

first_map    = Map(fits_files[0])
ref_observer = first_map.observer_coordinate

# Muestra para fijar dimensiones de píxeles
bl_s = SkyCoord(-CROP_LIM, -CROP_LIM, frame=first_map.coordinate_frame)
tr_s = SkyCoord( CROP_LIM,  CROP_LIM, frame=first_map.coordinate_frame)
sample = first_map.submap(bottom_left=bl_s, top_right=tr_s)
ny_ref, nx_ref = sample.data.shape
total = len(fits_files)
print(f"→ Cada recorte será {ny_ref}×{nx_ref} píxeles (ventana fija)")

def process_file(path, observer, out_dir, crop_lim, dims, idx, total):
    """
    Lee, rota, recorta, remuestrea y guarda un FITS.
    """
    fname = os.path.basename(path)
    prog  = (idx + 1) / total * 100
    print(f"[{idx+1}/{total}] {prog:.1f}% – {fname}")

    # 1) Leer y rotar diferencialmente
    smap  = Map(path)
    m_rot = differential_rotate(smap, observer=observer)

    # 2) Definir esquinas en coordenadas
    bl = SkyCoord(-crop_lim, -crop_lim, frame=m_rot.coordinate_frame)
    tr = SkyCoord( crop_lim,  crop_lim, frame=m_rot.coordinate_frame)

    # 3) Extraer submap helioprojectivo
    m_sub = m_rot.submap(bottom_left=bl, top_right=tr)

    # 4) Remuestrear a la forma fija en píxeles
    m_crop = m_sub.resample((dims[0], dims[1]) * u.pix)

    # 5) Conservar fecha original
    m_crop.meta['DATE-OBS'] = parse_time(m_rot.date).isot

    # 6) Guardar FITS
    out_path = os.path.join(out_dir, f"dr_crop_{fname}")
    m_crop.save(out_path, overwrite=True)

    return out_path

if __name__ == "__main__":
    # Número de procesos = núcleos de CPU
    workers = os.cpu_count() or 4
    dims    = (ny_ref, nx_ref)
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = []
        for i, f in enumerate(fits_files):
            futures.append(
                exe.submit(process_file,
                           f, ref_observer, OUTPUT_DIR, CROP_LIM, dims, i, total)
            )
        for fut in as_completed(futures):
            res = fut.result()
            print(res)

    print("Preprocesamiento paralelo finalizado. Salida en:", OUTPUT_DIR)
