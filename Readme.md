# Resumen del repositorio `solar-pmodes-removal`

Este repositorio contiene código y datos para descargar, preprocesar y filtrar modos p en imágenes HMI, así como experimentos con redes neuronales para acelerar el filtrado.

---

## 📂 Estructura general

solar-pmodes-removal/
├── images_intensity/
├── filtering_algorithm/
├── LICENSE
├── test_data/
└── training/


## 📂 `images_intensity/` 

Scripts para descargar, preprocesar, recortar y visualizar secuencias de intensidad continua (`Ic_45s`):

- **`intensity_dowload.py`**  
  Descarga de datos HMI Ic_45s, con cache local.

- **`preprocess.py`** y **`parallel_preprocess.py`**  
  Co‑alineación (differential rotation) y recorte a ±500 arcsec, en serie y en paralelo.

- **`visualize_crop_nocrop.py`**  
  Visualiza lado a lado imágenes originales y recortadas.

- **`len_verification.py`**  
  Verifica que todas las imágenes tengan la misma dimensión.

- **`bx_by_dim.py`**  
  Calcula las dimensiones de bloque para hacer luego la modificación correspondiente en `BigNFFT`.

- **`animate_gif.py`**  
  Crea GIFs de la evolución temporal de las secuencias.

---

## 📂 `filtering_algorithm/`

Contiene la implementación clásica del filtro subsonic (BigSonic) y utilidades para animar y probar el cubo:


- **`main.py`**  
  Script de ejemplo que construye el cubo, aplica `bigsonic()` y guarda `filtered_cube.npy`.

- **`bigsonic_hmi.py`**  
  Código principal que genera el filtro subsonic vía FFT 3D y aplica BigNFFT.

- **`bignfft_new.py`**  
  Clase `BigNFFT` para procesamiento en lotes y memmap, optimizada para cubos grandes.

- **`animation_cube.py`**  
  Genera una animación GIF del cubo filtrado.

- **`test.py`**  
  Pruebas básicas de consistencia y verificación rápida de la longitud de los datos antes del preprocesamiento.

- **`bigsonic_output/`**  
  Carpeta temporal donde `BigNFFT` escribe archivos intermedios.

---


---

## 📂 `test_data/`

- **`data_test.ipynb`**  
  Notebook de prueba con ejemplos mínimos de descarga, visualización y filtrado.

---

## 📂 `training/` 

Contiene los datos y scripts para entrenar y evaluar la red neuronal “1 a 1” y pruebas del PIML 3D:


- **`filter_verification.py`**  
  filter_verification.py crea el cubo de datos antes del filtrado, necesario para el entrenamiento de la red neuronal.
  Además, verifica la supresión de modos p tras pasar por la red (FFT temporal y cálculo de potencia).

- **`one_one_filtering_ml.py`**  
  Entrena y evalúa un autoencoder 2D que mapea cada imagen cruda → filtrada. Guarda el modelo entrenado. 

- **`ml_cube_generation.py`**  
  Script final de inferencia que recarga el modelo y genera `ml_cube.npy`.

- **`filter_after_ml.py`**
    Compara los filtrados y la imagen original para comprobar el funcionamiento de la red neuronal (mediante evaluación de la    transformada de Fourier)

- **`cube_ml_animation.py` (1.0 KB)**  
      Genera GIF del cubo filtrado por ML.

- **`many_times_filtering_ml.py`**  
  Modelo alternativo que aún se está estudiando.

---

## 📄 `LICENSE` (34 KB)

Licencia de uso del repositorio.

## ▶️ Flujo de trabajo típico

1. **Descarga y preprocesamiento** (`images_intensity/`):  
   - Co‑alinear, recortar y guardar FITS en `data_hmi_Ic_45s_crop_dr/`.
2. **Filtrado clásico** (`filtering_algorithm/main.py`):  
   - Genera `filtered_cube.npy` con BigSonic.
3. **Entrenamiento ML** (`training/one_one_filtering_ml.py`):  
   - Ajusta un autoencoder 2D en pares `(raw, filtered)`.
4. **Inferencia ML** (`training/ml_cube_generation.py`):  
   - Produce `ml_cube.npy` con la red, luego verifica con FFT (`filter_verification.py`).
---

