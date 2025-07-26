# Resumen del repositorio `solar-pmodes-removal`

Este repositorio contiene código y datos para descargar, preprocesar y filtrar modos p en imágenes HMI, así como experimentos con redes neuronales para acelerar el filtrado.

---

## 📂 Estructura general

solar-pmodes-removal/
├── filtering_algorithm/
├── images_intensity/
├── LICENSE
├── test_data/
└── training/


---

## 📂 `filtering_algorithm/` (≈ 6.9 MB)

Contiene la implementación clásica del filtro subsonic (BigSonic) y utilidades para animar y probar el cubo:

- **`bigsonic_hmi.py` (6.3 KB)**  
  Código principal que genera el filtro subsonic vía FFT 3D y aplica BigNFFT.

- **`bignfft_new.py` (4.7 KB)**  
  Clase `BigNFFT` para procesamiento en lotes y memmap, optimizada para cubos grandes.

- **`main.py` (1.3 KB)**  
  Script de ejemplo que construye el cubo, aplica `bigsonic()` y guarda `filtered_cube.npy`.

- **`animation_cube.py` (989 B)**  
  Genera una animación GIF del cubo original y filtrado.

- **`test.py` (622 B)**  
  Pruebas básicas de consistencia y verificación rápida.

- **`bigsonic_output/`**  
  Carpeta temporal donde `BigNFFT` escribe archivos intermedios.

- **`filtered_cube.gif` (6.9 MB)**  
  GIF de demostración con la evolución del cubo después del filtrado.

---

## 📂 `images_intensity/` (≈ 1.3 MB)

Scripts para descargar, preprocesar, recortar y visualizar secuencias de intensidad continua (`Ic_45s`):

- **`intensity_dowload.py` (2.2 KB)**  
  Descarga de datos HMI Ic_45s, con cache local.

- **`preprocess.py` (2.0 KB)** y **`parallel_preprocess.py` (2.7 KB)**  
  Co‑alineación (differential rotation) y recorte a ±500 arcsec, en serie y en paralelo.

- **`visualize_crop_nocrop.py` (1.5 KB)**  
  Visualiza lado a lado imágenes originales y recortadas.

- **`len_verification.py` (583 B)**  
  Verifica que todas las imágenes tengan la misma dimensión.

- **`bx_by_dim.py` (2.3 KB)**  
  Calcula las dimensiones de bloque para `BigNFFT`.

- **`animate_gif.py` (2.2 KB)**  
  Crea GIFs de la evolución temporal de las secuencias.

- **`cropped.gif` (1.3 MB)**  
  Animación del cubo recortado.

---

## 📂 `test_data/` (≈ 10 KB)

- **`data_test.ipynb` (6.3 KB)**  
  Notebook de prueba con ejemplos mínimos de descarga, visualización y filtrado.

---

## 📂 `training/` (≈ 196 MB)

Contiene los datos y scripts para entrenar y evaluar la red neuronal “1 a 1” y pruebas del PIML 3D:

- **`cube_raw.npy` (196 MB)**  
  Cubo original de entrenamiento (sin filtrar).

- **`one_one_filtering_ml.py` (3.8 KB)**  
  Entrena y evalúa un autoencoder 2D que mapea cada imagen cruda → filtrada.

- **`many_times_filtering_ml.py` (3.6 KB)**  
  Aplica el modelo 2D a todo el cubo y guarda `ml_cube.npy`.

- **`ml_cube_generation.py` (1.6 KB)**  
  Script final de inferencia que recarga el modelo y genera `ml_cube.npy`.

- **`filter_after_ml.py` (2.0 KB)** y **`filter_verification.py` (2.3 KB)**  
  Verifican la supresión de modos p tras pasar por la red (FFT temporal y cálculo de potencia).

- **`cube_ml_animation.py` (1.0 KB)**  
  Genera GIF comparando cubo filtrado clásico vs. ML.

---

## 📄 `LICENSE` (34 KB)

Licencia de uso del repositorio.

---

