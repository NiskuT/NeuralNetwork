# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:46:55 2020

@author: qjupi
"""

import time as t
start = t.time()




from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
"""
def convWithSingleFilter(self, img, filter1):

        out = np.zeros(((img.shape[0]-filter1.shape[0]+1),(img.shape[1]-filter1.shape[1]+1)))
        
        for k in range(out.shape[0]):
            for i in range(out.shape[1]):
                out[k,i]+=np.sum(img[k:k+filter1.shape[0], i:i+filter1.shape[1]]*filter1)
                
                
        return out
"""

@cuda.jit
def convWithSingleFilter(I,F,O):
    x,y = cuda.grid(2)
    if(x<O.shape[0] and y<O.shape[1]):
        seum=0
        for i in range(3):
            for j in range(3):
                seum += F[i,j] * I[x+i-1,y+j-1]
        O[x,y]=seum
    
    
im = plt.imread("plage.jpg")
IMG = np.ascontiguousarray(im[:,:,0])
filtre = np.array([[0,1,0],[1,-1,1],[0,1,0]])

IMGGlobal = cuda.to_device(IMG)
filtreGlobal = cuda.to_device(filtre)
outx,outy=IMG.shape-np.array((2,2))
sortieGlobal = cuda.device_array((outx,outy))

threadParBlock = (32,32)
blocksParGrille = (int(ceil(outx/32)),int(ceil(outy/32)))

start = t.time()
convWithSingleFilter[blocksParGrille, threadParBlock](IMGGlobal,filtreGlobal,sortieGlobal)
end = t.time()
print("Elapsed = %s" % (end - start))


out = sortieGlobal.copy_to_host()

plt.imshow(out)
"""

@cuda.jit
def convWithSingleFilter(I,F,O):
    tx,ty = cuda.threadIdx.x,cuda.threadIdx.y
    tblockx, tblocky = cuda.blockIdx.x,cuda.blockIdx.y
    
    x,y=tx+tblockx%(cuda.blockDim.x//3),ty+tblocky%(cuda.blockDim.y//3)
    i,j = tblockx//(cuda.blockDim.x//3),tblocky//(cuda.blockDim.y//3)
    
    if(x<O.shape[0] and y<O.shape[1]):
        
        O[x,y]+= I[x+i,y+j]*F[i,j]
    
    
im = plt.imread("plage.jpg")
IMG = np.ascontiguousarray(im[:,:,0])
filtre = np.array([[0,1,0],[1,-1,1],[0,1,0]])

IMGGlobal = cuda.to_device(IMG)
filtreGlobal = cuda.to_device(filtre)
outx,outy=IMG.shape-np.array((2,2))
sortieGlobal = cuda.device_array((outx,outy))

threadParBlock = (32,32)
blocksParGrille = (3*int(ceil(outx/32)),3*int(ceil(outy/32)))

start = t.time()
convWithSingleFilter[blocksParGrille, threadParBlock](IMGGlobal,filtreGlobal,sortieGlobal)
end = t.time()
print("Elapsed = %s" % (end - start))


out = sortieGlobal.copy_to_host()

plt.imshow(out)
"""