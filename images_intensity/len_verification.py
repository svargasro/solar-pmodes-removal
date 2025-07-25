#!/usr/bin/env python3

import os
import glob
import numpy as np
from sunpy.map import Map

INPUT_DIR = "./data_hmi_Ic_45s_crop_dr/"

# -------- LEER IMÁGENES Y CREAR CUBO --------
fits_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.fits")))
if not fits_files:
    raise FileNotFoundError(f"No FITS files found in {INPUT_DIR}")

initial_len = 0
for index, f in enumerate(fits_files):
    if index==0:
        initial_len= len(Map(f).data)
    if initial_len != len(Map(f).data):
        print(f)
        print(len(Map(f).data))
    print("Avance: ", index*100/len(fits_files))
