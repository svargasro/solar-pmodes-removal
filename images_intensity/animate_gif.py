#!/usr/bin/env python3
# animate_to_gif.py

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sunpy.map import Map
from astropy.time import Time

# -------- CONFIGURACIÓN --------
INPUT_DIR = "data_hmi_Ic_45s/"
CROP_DIR  = "data_hmi_Ic_45s_crop_dr/"
FPS       = 4
DURATION  = None
PALETTE   = "gray"

# -------- AUX: obtener lista ordenada de FITS --------
def get_fits(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.fits")))
    if not files:
        raise FileNotFoundError(f"No FITS found in {directory}")
    return files

# -------- PARSE FECHA DESDE NOMBRE --------


def parse_date_from_filename(fname):
    try:
        base = os.path.basename(fname)
        start = base.find("2023_")
        date_str = base[start:start + 19]  # '2023_06_01_00_01_30'
        # Convertir a formato ISO válido: 2023-06-01T00:01:30
        date_str = date_str.replace("_", "-", 2).replace("_", "T", 1).replace("_", ":", 2)
        return Time(date_str, format="isot")
    except Exception as e:
        print(f"ERROR al parsear fecha desde nombre: {fname}")
        raise e

# -------- CREAR GIF --------
def create_gif(fits_files, title, output_path):
    maps = [Map(f) for f in fits_files]
    dates = [parse_date_from_filename(f) for f in fits_files]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(projection=maps[0].wcs)
    img = ax.imshow(maps[0].data, origin='lower', cmap=PALETTE)
    cbar = fig.colorbar(img, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Intensidad [DN]')

    def update(i):
        img.set_data(maps[i].data)
        ax.set_title(f"{title} - {dates[i].strftime('%Y-%m-%d %H:%M:%S')}")
        return img,

    nframes = len(maps)
    writer = animation.PillowWriter(fps=FPS if not DURATION else int(nframes / DURATION))
    ani = animation.FuncAnimation(fig, update, frames=nframes, blit=True)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print(f"GIF guardado en: {output_path}")

if __name__ == '__main__':
    full_files = get_fits(INPUT_DIR)
    crop_files = get_fits(CROP_DIR)

    create_gif(crop_files, "Cropped", "cropped.gif")
    create_gif(full_files, "Full Disk", "full_disk.gif")

    print("Animaciones GIF generadas.")
