#!/usr/bin/env python3

#!/usr/bin/env python3
# apply_bigsonic.py

import os
import glob
import numpy as np
from sunpy.map import Map
from bignfft_new import BigNFFT
from bigsonic_hmi import bigsonic  # Renombra tu script original a bignfft_script.py
                                     # y asegúrate de que defina la función bigsonic

# -------- CONFIGURACIÓN --------
INPUT_DIR = "./../images_intensity/data_hmi_Ic_45s_crop_dr/"
OUTPUT_PATH = "bigsonic_output/"
BXDIM = 1001.5274800000001 #216
BYDIM = 1001.5274800000001 #216

# -------- LEER IMÁGENES Y CREAR CUBO --------
fits_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.fits")))
if not fits_files:
    raise FileNotFoundError(f"No FITS files found in {INPUT_DIR}")

for f in fits_files:
    print(len(Map(f).data))
# Cargar los mapas como un cubo de datos: (tiempo, altura, ancho)
cube_data = np.stack([Map(f).data for f in fits_files])
first_index = 0
last_index = cube_data.shape[0] - 1

# -------- APLICAR FILTRO --------
filtered_cube = bigsonic(
    cube=cube_data,
    first=first_index,
    last=last_index,
    bxdim=BXDIM,
    bydim=BYDIM,
    path_tmp=OUTPUT_PATH
)

# -------- GUARDAR CUBO FILTRADO --------
np.save("filtered_cube.npy", filtered_cube)
print("Cubo filtrado guardado como 'filtered_cube.npy'")
