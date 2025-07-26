#!/usr/bin/env python3

#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parámetros
CUBE_PATH = "ml_cube.npy"
OUTPUT_GIF = "ml_cube.gif"
FPS = 5  # Cambia si deseas que vaya más rápido o más lento

# Cargar cubo filtrado
cube = np.load(CUBE_PATH)
nframes = cube.shape[0]

# Crear figura
fig, ax = plt.subplots(figsize=(6, 6))
img = ax.imshow(cube[0], origin="lower", cmap="gray", animated=True)
cbar = plt.colorbar(img, ax=ax)
cbar.set_label("Intensidad [DN]")
ax.set_title("Cubo filtrado")

# Función de actualización para cada frame
def update(i):
    img.set_array(cube[i])
    ax.set_title(f"Cubo filtrado – Frame {i+1}/{nframes}")
    return img,

# Crear animación
ani = animation.FuncAnimation(fig, update, frames=nframes, interval=1000/FPS, blit=True)

# Guardar como GIF
ani.save(OUTPUT_GIF, writer=animation.PillowWriter(fps=FPS))
plt.close(fig)

print(f"GIF guardado como '{OUTPUT_GIF}' con {nframes} frames.")
