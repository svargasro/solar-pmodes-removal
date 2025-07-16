#!/usr/bin/env python
# -*- coding: utf-8 -*-


from astropy.io import fits
import numpy as np
import os
import time
from bignfft_new import *
import shutil
import glob
#~ import sunpy.map as smap
import sys
from bignfft_new import BigNFFT
from tqdm import tqdm


print("Iniciando codigo bigsonic")

scale = 1814.544 #arcsec/pixel ; change manually
t_step = 45 #Mean time separation between images [s] ; change manually
v_ph = 4.0 # Maximum phase velocity [km/s]

def bigsonic(cube,first,last,bxdim,bydim,path_tmp):
    if not os.path.exists(path_tmp+"work"):
        os.makedirs(path_tmp+"work")
    if not os.path.exists(path_tmp+"filter"):
        os.makedirs(path_tmp+"filter")
        
    str0 = path_tmp+"work/"
    strf = path_tmp+"filter/"

    
    # lineas editadas para que funcione con el cubo generado por Sunpy
    
    print("Cargando Cubo")

    
    # ====================================================
    # Working with cube data made by pyfits or astropy

    dimx = cube.shape[2]
    dimy = cube.shape[1]
    # ===================================================
    
    
    
    xdim=dimx
    ydim=dimy
    x_anf=0
    y_anf=0
    
    
#-----------------------------------------------------------------

    if xdim%2 != 0: xdim = xdim - 1    
    if ydim%2 != 0: ydim = ydim - 1    

    if (last - first + 1)%2 != 0: last = last - 1    

    tdim = last - first + 1

    print("-------")
    
    
    
    ap = 0
    
    cut = 0
    
    if (cut > 2) or (cut < 0):
        raise ValueError('The cut values alowed are 0, 1, and 2')
    if (ap != 0) or (cut != 0):
        perct = 10
        smooth_t = int(tdim*perct/100) # width of the edge in t-dimension

    t1 = time.time()
    
    #-----------------------------------------
    
    print(tdim, " images to be filtered")
    
    # subsonic construction
    
    # Definin' unit in the Fourier domain
    
    print("Spatial resolution ->", scale, " arcsec/pixel")
    
    print("---")
    
    kx_step = 1./(scale*725.*xdim)
    ky_step = 1./(scale*725.*ydim)
    w_step = 1./(t_step*tdim)
    
    
    # Prepare de filter
    
    nx = int(xdim/2) + 1
    ny = int(ydim/2) + 1
    nt = int(tdim/2) + 1
    filter_mask = np.zeros([ny,nx,nt],dtype=float)
    
    
    if (cut == 0) or (cut == 1):
        print("Now calculatin' filter...")
        for j in range(ny):
            for i in range(nx):
                k_by_v = np.sqrt((i*kx_step)**2+(j*ky_step)**2)*v_ph
                for k in range(1,nt):
                    if k*w_step <= k_by_v:
                        filter_mask[j,i,k]=1.
    
    if cut == 1:
        for i in range(nx):
            for j in range(ny):
                for k in range(1,nt):
                    if filter_mask[j,i,k] != 1:
                        trans = perct*k/100
                        trans2 = int(trans/2)
                        trans = 2*trans
                        if trans2 >= 1:
                            n=0
                            for kk in range(int(k-trans2),int(k+trans2)):
                                if kk <= nt-1:
                                    filter_mask[j,i,kk] = 0.5+0.5*np.cos(np.pi*n/trans)
                                    n = n+1
                                else:
                                    continue
    

    
    for i in range(nt):
        filter_slice = np.zeros([ydim,xdim])
        if i == 0:
            filter_slice = filter_slice + 1.
        else:
            filter_slice[0:ny,0:nx]=filter_mask[:,:,i]
            filter_slice[ny:ydim,:]=filter_slice[1:ny-1,:][::-1,:]
            filter_slice[:,nx:xdim]=filter_slice[:,1:nx-1][:,::-1]
        dcn = str(i).zfill(4)
        np.save(strf+dcn+".npy",filter_slice)
        
        if i != 0:
            dcn = str(tdim-i+first).zfill(4)
            np.save(strf+dcn+".npy",filter_slice)

    print("The filter was written to "+strf)
    
    del(filter_mask)
    del(filter_slice)
    
    
    if ap != 0:
        tmask = np.ones(tdim)
        for i in range(smooth_t):
            tmask[i] = (1-np.cos(np.pi*i/smooth_t))/2
        tmask[tdim-smooth_t:tdim] = (tmask[1:smooth_t+1])[::-1]
        print('Computing the mean for the cube (it could take several minutes):')
        av = 0.
        for n in range(first,last+1):
            ima = cube[n,:,:]
            ima = ima[y_anf:y_anf+ydim,x_anf:x_anf+xdim]
            av = av+np.mean(ima)/tdim
    
    
    # Loop of reading, optional apodization and writing images
    
    print('Reading apodization')
    for n in tqdm(range(first,last+1)):
        dcn = str(n).zfill(4)
        
        # Editado para trabajar con sunpy
        
        ima = cube[n,:,:]
        ima = ima.astype('float32')
        
        # apodization
        if ap != 0:
            ima = ima-av
            ima = ima*tmask[n-first]
            ima = ima + av
            del(tmask)
        

        np.save(str0+"apo"+dcn+".npy",ima)

    
    del(ima)
    
    
    # Direct FFT
    
    print(50*"=")
    print("Calling bignfft")
    print(50*"-")
    
    nfft_processor = BigNFFT(dimx, dimy, bxdim, bydim, path_tmp, batch_size=8)
    nfft_processor.run(pm=-1, first=first, last=last)
    #bignfft(-1, first,last,dimx,dimy,bxdim,bydim,path_tmp)

    
    # Multiplying transformed images by filter images
    
    print('applying filter')
    
    
    for n in tqdm(range(first,last+1)):
        dcn = str(n).zfill(4)
        ima = np.load(str0+"fft"+dcn+".npy")
        filter = np.load(strf+dcn+".npy")
        
        
        ima = ima*filter
        np.save(str0+"fft"+dcn+".npy",ima)

    del(ima)
    del(filter)
    
    #Inverse FFT
    print(50*"=")
    print("Calling bignfft")
    print(50*"-")
    
    #bignfft(1, first,last,dimx,dimy,bxdim,bydim,path_tmp)
    nfft_processor.run(pm=1, first=first, last=last)

    
    # Saving results
    
    ima = np.zeros([ydim,xdim],dtype="complex64")
    
    cube_new = np.zeros([tdim,ydim,xdim],dtype='float32')

    
    
    for n in range(first,last+1):
        dcn = str(n).zfill(4)
        ima = np.load(str0+"F"+dcn+".npy")
        im = (ima.real)
        cube_new[n,:,:] = im

    
    print("---")
    print("Total elapsed time from begining = ", np.round(time.time()-t1,2))
    print(" ")
    print("Erasing directories work and filter")
    
    shutil.rmtree(str0)
    shutil.rmtree(strf)
    return cube_new


        

    
        
    
        
    
    
    
    
    
    
