#!/usr/bin/env python3

import glob
import os
from astropy.io import fits

INPUT_DIR = "./data_hmi_Ic_45s_crop_dr"

# 1. Buscar archivos FITS
fits_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.fits")))
if not fits_files:
    raise FileNotFoundError(f"No se encontraron archivos FITS en {INPUT_DIR}")

# 2. Mostrar lista numerada de archivos
print("Archivos FITS encontrados:")
# for idx, file_path in enumerate(fits_files):
#     print(f"{idx}: {os.path.basename(file_path)}")

# 3. Pedir al usuario que seleccione un archivo
selected_idx = 0
if selected_idx < 0 or selected_idx >= len(fits_files):
    raise ValueError("Índice inválido")

selected_file = fits_files[selected_idx]

# 4. Leer el archivo FITS
with fits.open(selected_file) as hdul:
    print(f"\nAnálisis de {os.path.basename(selected_file)}")
    print(f"El archivo contiene {len(hdul)} extensiones HDUs")

    # 5. Permitir selección manual de HDU
    hdu_choice = 0
    if hdu_choice < 0 or hdu_choice >= len(hdul):
        raise ValueError("Número de HDU inválido")

    header = hdul[hdu_choice].header

    # 6. Calcular dimensiones equivalentes si existen los parámetros necesarios
    print("\nResultados:")
    if all(key in header for key in ['NAXIS1', 'NAXIS2', 'CDELT1', 'CDELT2']):
        bxdim_equiv = header['NAXIS1'] * header['CDELT1']
        bydim_equiv = header['NAXIS2'] * header['CDELT2']

        print(f"  NAXIS1: {header['NAXIS1']} píxeles")
        print(f"  NAXIS2: {header['NAXIS2']} píxeles")
        print(f"  CDELT1: {header['CDELT1']} arcsec/píxel")
        print(f"  CDELT2: {header['CDELT2']} arcsec/píxel")
        print(f"  BXDIM calculado: {bxdim_equiv} arcsec")
        print(f"  BYDIM calculado: {bydim_equiv} arcsec")
    else:
        print("  No se encontraron todos los parámetros necesarios (NAXIS1, NAXIS2, CDELT1, CDELT2) para el cálculo")

    # 7. Mostrar BXDIM/BYDIM directos si existen
    bxdim = header.get('BXDIM', None)
    bydim = header.get('BYDIM', None)
    if bxdim is not None or bydim is not None:
        print(f"  BXDIM directo: {bxdim if bxdim is not None else 'No encontrado'}")
        print(f"  BYDIM directo: {bydim if bydim is not None else 'No encontrado'}")

    # 8. Opción para mostrar todas las cabeceras
    if False:
        print("\nCabeceras completas:")
        for key, value in header.items():
            print(f"  {key}: {value}")
