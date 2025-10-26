# Summary of the `solar-pmodes-removal` Repository

This repository contains code and data to download, preprocess, and filter **p-modes** in HMI images, as well as experiments with neural networks to accelerate the filtering process.

---

## 📂 General Structure

solar-pmodes-removal/
├── images_intensity/
├── filtering_algorithm/
├── training/
├── LICENSE
└── test_data/

---

## 📂 `images_intensity/`

Scripts for downloading, preprocessing, cropping, and visualizing HMI continuum intensity (`Ic_45s`) sequences:

- **`intensity_download.py`**  
  Downloads HMI Ic_45s data with local caching.

- **`preprocess.py`** and **`parallel_preprocess.py`**  
  Co-alignment (differential rotation) and cropping to ±500 arcsec, sequential and parallel versions.

- **`visualize_crop_nocrop.py`**  
  Displays side-by-side original and cropped images.

- **`len_verification.py`**  
  Verifies that all images have the same dimensions.

- **`bx_by_dim.py`**  
  Computes block dimensions to later adjust `BigNFFT` configuration.

- **`animate_gif.py`**  
  Creates GIFs showing the temporal evolution of the sequences.

---

## 📂 `filtering_algorithm/`

Contains the classic implementation of the **subsonic filter** (BigSonic) and utilities to animate and test the data cube:

- **`main.py`**  
  Example script that builds the cube, applies `bigsonic()`, and saves `filtered_cube.npy`.

- **`bigsonic_hmi.py`**  
  Main code that generates the subsonic filter via 3D FFT and applies `BigNFFT`.

- **`bignfft_new.py`**  
  `BigNFFT` class for batch and memmap processing, optimized for large cubes.

- **`animation_cube.py`**  
  Generates an animated GIF of the filtered cube.

- **`test.py`**  
  Basic consistency checks and quick verification of data length before preprocessing.

- **`bigsonic_output/`**  
  Temporary folder where `BigNFFT` writes intermediate files.

---

## 📂 `test_data/`

- **`data_test.ipynb`**  
  Test notebook with minimal examples for downloading, visualizing, and filtering data.

---

## 📂 `training/`

Contains data and scripts to train and evaluate the **1-to-1 neural network** and 3D PIML tests:

- **`filter_verification.py`**  
  Builds the pre-filtered data cube required for training and verifies p-mode suppression after passing through the neural network (temporal FFT and power spectrum analysis).

- **`one_one_filtering_ml.py`**  
  Trains and evaluates a 2D autoencoder mapping raw → filtered images. Saves the trained model.

- **`ml_cube_generation.py`**  
  Inference script that loads the model and generates `ml_cube.npy`.

- **`filter_after_ml.py`**  
  Compares filtered and original images to check the neural network performance (via Fourier transform analysis).

- **`cube_ml_animation.py` (1.0 KB)**  
  Generates GIF animations of the ML-filtered cube.

- **`many_times_filtering_ml.py`**  
  Alternative model currently under study.

---

## 📄 `LICENSE` (34 KB)

Repository usage license.

---

## ▶️ Typical Workflow

1. **Download and preprocessing** (`images_intensity/`):  
   - Co-align, crop, and save FITS files in `data_hmi_Ic_45s_crop_dr/`.

2. **Classic filtering** (`filtering_algorithm/main.py`):  
   - Generates `filtered_cube.npy` using BigSonic.

3. **ML training** (`training/one_one_filtering_ml.py`):  
   - Trains a 2D autoencoder on `(raw, filtered)` pairs.

4. **ML inference** (`training/ml_cube_generation.py`):  
   - Produces `ml_cube.npy` using the trained network, then verifies with FFT (`filter_verification.py`).

---
