# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:46:55 2020

@author: qjupi
"""

import time as t

from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def convSingle(img, filter1):

        out = np.zeros(((img.shape[0]-filter1.shape[0]+1),(img.shape[1]-filter1.shape[1]+1)))
        
        for k in range(out.shape[0]):
            for i in range(out.shape[1]):
                out[k,i]+=np.sum(img[k:k+filter1.shape[0], i:i+filter1.shape[1]]*filter1)
                
                
        return out
        
        
def convolution(imageBank, filterBank):
        

        output = np.zeros((imageBank.shape[0]-filterBank.shape[0]+1, \
                           imageBank.shape[1]-filterBank.shape[1]+1, \
                           imageBank.shape[2] * filterBank.shape[2]  ))
        
        for image in range( imageBank.shape[2] ):
            for filtre in range( filterBank.shape[2] ):
               
                output[:,:,filterBank.shape[2]*image+filtre] = convSingle(imageBank[:,:,image], filterBank[:,:,filtre])
                
        return output


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
def conv(I,F,O):
    tx,ty,tz = cuda.threadIdx.x,cuda.threadIdx.y,cuda.threadIdx.z
    tblockx, tblocky,tblockz = cuda.blockIdx.x,cuda.blockIdx.y,cuda.blockIdx.z
    
    x,y,z=tx+(tblockx%(cuda.gridDim.x//3)*cuda.blockDim.x),ty+(tblocky%(cuda.gridDim.y//3)*cuda.blockDim.y),tz+tblockz*cuda.blockDim.z
    #cuda.gridDim.x//3 -> taille "réelle" de l'image (//3 car filtre 3x3)
    #tblockx % ^^^^^pour obtenir le num du block dans l'image
    
    i,j = tblockx//(cuda.gridDim.x//3),tblocky//(cuda.gridDim.y//3)
    # tblockx // ^^^^^^^pour obtenir "la case du filtre correspondante
    
    if(x<O.shape[0] and y<O.shape[1] and z<O.shape[2]):
        
        O[x,y,z] += I[x+i,y+j,z//F.shape[2]]*F[i,j,z%F.shape[2]]
    
 
im = plt.imread("plage.jpg")
IMG = np.ascontiguousarray(im)
filterL = [np.array([[0,1,0],[1,-1,1],[0,1,0]]), np.array([[-1,-1,-1],[0,0,0],[1,1,1]]), np.array([[0,1,0],[1,-1,1],[0,1,0]])]
filtre = np.ascontiguousarray(np.transpose(np.array(filterL), axes=(1,2,0)))


IMGGlobal = cuda.to_device(IMG)
filtreGlobal = cuda.to_device(filtre)

outx,outy,outz = IMG.shape[0]-filtre.shape[0]+1, IMG.shape[1]-filtre.shape[1]+1, IMG.shape[2] * filtre.shape[2]

sortieGlobal = cuda.device_array(( outx,outy,outz ))

threadParBlock = (16,16,4)
blocksParGrille = (3*int(ceil(outx/16)),3*int(ceil(outy/16)), int(ceil(outz/4)))

start = t.time()
conv[blocksParGrille, threadParBlock](IMGGlobal,filtreGlobal,sortieGlobal)
end = t.time()
print("Temps = %s" % (end - start))

start = t.time()
out2 = convolution(IMG,filtre)
end = t.time()
print("Temps = %s" % (end - start))


out = sortieGlobal.copy_to_host()

plt.imshow(out[:,:,3])
