#!/usr/bin/env python3
# intensity_download.py

import os
import glob
from sunpy.net import Fido, attrs as a
from astropy import units as u
from astropy.io import fits
import sunpy.map
from sunpy.time import parse_time
import numpy as np
import matplotlib.pyplot as plt

# -------- PARÁMETROS --------
fecha    = "2023-06-01"
hora_ini = "00:00"
hora_fin = "00:10"
cadencia = 45 * u.second
carpeta  = "data_hmi_Ic_45s/"  # carpeta para intensidad de continuo

# -------- CREA LA CARPETA SI NO EXISTE --------
os.makedirs(carpeta, exist_ok=True)

# -------- BUSCA ARCHIVOS LOCALMENTE --------
files = sorted(glob.glob(os.path.join(carpeta, "*.fits")))

# -------- SI NO HAY ARCHIVOS, DESCÁRGALOS VIA VSO --------
if len(files) == 0:
    print("No se encontraron archivos. Descargando intensidad de continuo vía VSO ...")
    result = Fido.search(
        a.Time(f"{fecha} {hora_ini}", f"{fecha} {hora_fin}"),
        a.Instrument.hmi,
        a.Physobs.intensity,
        a.Sample(cadencia)
    )
    files = Fido.fetch(result, path=carpeta + "{file}")
else:
    print(f"{len(files)} archivos encontrados localmente.")

# -------- GRAFICA LA PRIMERA IMAGEN DE INTENSIDAD --------
smap = sunpy.map.Map(files[1])

plt.figure(figsize=(8, 6))
smap.plot(cmap="gray")
plt.colorbar(label="Intensidad [DN]")
plt.title("Continuum Intensity (HMI Ic) — 45 s")
plt.show()

# -------- OPCIONAL: FFT DE LA SEÑAL EN EL PÍXEL CENTRAL --------
# Para descomentarlo, quita los hashes de las próximas líneas

# times = []
# values = []
# for f in files:
#     try:
#         with fits.open(f) as hdu:
#             data = hdu[1].data
#             values.append(data[512, 512])
#             times.append(parse_time(hdu[1].header["DATE-OBS"]).unix)
#     except Exception as e:
#         print(f"Error al leer {f}: {e}")

# signal = np.array(values) - np.mean(values)
# freqs = np.fft.rfftfreq(len(signal), d=cadencia.to("s").value)
# spectrum = np.abs(np.fft.rfft(signal))

# plt.figure(figsize=(8,5))
# plt.plot(freqs * 1e3, spectrum)
# plt.axvline(3.3, color='r', linestyle='--',
#             label='p-mode (~3.3 mHz)')
# plt.xlabel("Frecuencia [mHz]")
# plt.ylabel("Amplitud")
# plt.title("FFT de intensidad — píxel central")
# plt.legend()
# plt.grid(True)
# plt.show()
