from pathlib import Path
import numpy as np
from scipy.fft import fft2, ifft2, fft, ifft
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import gc
import os

class BigNFFT:
    """
    Efficient N-dimensional FFT processor for large image cubes.
    """
    def __init__(self, dimx, dimy, bxdim, bydim, path_tmp, batch_size=8):
        self.dimx = dimx
        self.dimy = dimy
        self.bxdim = bxdim
        self.bydim = bydim
        self.path_tmp = Path(path_tmp)
        self.batch_size = batch_size
        self.str0 = self.path_tmp / "work"
        self.str0.mkdir(parents=True, exist_ok=True)

    def _process_subarray(self, args):
        cube, pm, miniy, maxiy, minix, maxix = args
        box3d = cube[:, miniy:maxiy, minix:maxix]
        if pm == -1:
            box3d = fft(box3d, axis=0)
        elif pm == 1:
            box3d = ifft(box3d, axis=0)
        cube[:, miniy:maxiy, minix:maxix] = box3d

    def run(self, pm, first, last):
        """
        Run the FFT/iFFT pipeline on the image cube.
        """
        xdim, ydim = self.dimx, self.dimy
        bxdim, bydim = self.bxdim, self.bydim
        batch_size = self.batch_size

        if xdim % 2 != 0: xdim -= 1
        if ydim % 2 != 0: ydim -= 1

        # Step 1: Batch FFT/iFFT
        print('Reading images')
        indices = np.arange(first, last + 1)
        for batch_start in tqdm(range(0, len(indices), batch_size)):
            batch_indices = indices[batch_start:batch_start + batch_size]
            ims = []
            for i in batch_indices:
                dcn = str(i).zfill(4)
                try:
                    if pm == -1:
                        im = np.load(self.str0 / f"apo{dcn}.npy")
                        ims.append(im[:ydim, :xdim])
                    elif pm == 1:
                        im = np.load(self.str0 / f"fft{dcn}.npy")
                        ims.append(im[:ydim, :xdim])
                except Exception as e:
                    print(f"Error loading {dcn}: {e}")
            ims = np.stack(ims)
            if pm == -1:
                fims = fft2(ims, axes=(-2, -1), norm="ortho").astype(np.complex64)
                for idx, i in enumerate(batch_indices):
                    dcn = str(i).zfill(4)
                    np.save(self.str0 / f"fft{dcn}.npy", fims[idx])
            elif pm == 1:
                fims = ifft2(ims, axes=(-2, -1), norm="ortho").astype(np.complex64)
                for idx, i in enumerate(batch_indices):
                    dcn = str(i).zfill(4)
                    np.save(self.str0 / f"F{dcn}.npy", fims[idx])
            del ims, fims
            gc.collect()

        # Step 2: Use memmap for the cube
        tdim = last - first + 1
        cube_path = self.str0 / "cube_memmap.dat"
        cube = np.memmap(cube_path, dtype=np.complex64, mode='w+', shape=(tdim, ydim, xdim))
        print("Loading all FFT images into memmap...")
        for idx, i in enumerate(range(first, last + 1)):
            dcn = str(i).zfill(4)
            if pm == -1:
                cube[idx] = np.load(self.str0 / f"fft{dcn}.npy")
            elif pm == 1:
                cube[idx] = np.load(self.str0 / f"F{dcn}.npy")
        cube.flush()
        gc.collect()

        # Step 3: Process subarrays in memory (parallel)
        a = xdim % bxdim
        b = ydim % bydim
        n = xdim // bxdim
        m = ydim // bydim
        dix = np.full(n + (1 if a != 0 else 0), bxdim, dtype=int)
        if a != 0: dix[-1] = a
        diy = np.full(m + (1 if b != 0 else 0), bydim, dtype=int)
        if b != 0: diy[-1] = b
        nelx, nely = len(dix), len(diy)
        print("Number of subarrays -->", nelx * nely)
        print('Processing subarrays in Fourier domain...')
        tasks = []
        for jbox in range(nely):
            for ibox in range(nelx):
                miniy = jbox * bydim
                maxiy = miniy + diy[jbox]
                minix = ibox * bxdim
                maxix = minix + dix[ibox]
                tasks.append((cube, pm, miniy, maxiy, minix, maxix))
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(self._process_subarray, tasks), total=len(tasks)))
        cube.flush()
        gc.collect()

        # Step 4: Save updated FFT images back to disk
        print("Saving updated FFT images...")
        for idx, i in enumerate(range(first, last + 1)):
            dcn = str(i).zfill(4)
            if pm == -1:
                np.save(self.str0 / f"fft{dcn}.npy", cube[idx])
            elif pm == 1:
                np.save(self.str0 / f"F{dcn}.npy", cube[idx])
        del cube
        gc.collect()
        # Optionally, remove the memmap file after use
        try:
            os.remove(cube_path)
        except Exception as e:
            print(f"Could not remove memmap file: {e}")