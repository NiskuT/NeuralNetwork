# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import time as t

from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
from math import ceil


def convBackprop(grad,filtre,inp):
    

    #inpGrad = np.zeros(inp.shape)
    filterGrad = np.zeros(filtre.shape)
    

    
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            for k in range(grad.shape[2]):

                filterGrad[:,:,k%filterGrad.shape[2]] += grad[i,j,k] * inp[i:i+filterGrad.shape[0],j:j+filterGrad.shape[1],k//filtre.shape[2]]
                #inpGrad[i:i+filterGrad.shape[0],j:j+filterGrad.shape[1],k//filterGrad.shape[2]] += grad[i,j,k]*filtre[k%filterGrad.shape[2]]
                
    #filtre -= filterGrad
    
    return filterGrad

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

#IMG = np.ascontiguousarray(np.copy(im))
IMG = np.random.randint(2, size=(5,5,2))
filterL = [np.array([[0,1,0],[1,-1,1],[0,1,0]]),  np.array([[0,1,0],[1,-1,1],[0,1,0]])]
filtre = np.ascontiguousarray(np.transpose(np.array(filterL), axes=(1,2,0)))


IMGGlobal = cuda.to_device(IMG)        # ENTREE 1
filtreGlobal = cuda.to_device(filtre)  # ENTREE 2

outx,outy,outz = IMG.shape[0]-filtre.shape[0]+1, IMG.shape[1]-filtre.shape[1]+1, IMG.shape[2] * filtre.shape[2]

sortieGlobal = cuda.device_array(( outx,outy,outz ))

threadParBlock = (16,16,4)
blocksParGrille = (3*int(ceil(outx/16)),3*int(ceil(outy/16)), int(ceil(outz/4)))

start = t.time()
conv[blocksParGrille, threadParBlock](IMGGlobal,filtreGlobal,sortieGlobal)
end = t.time()
print("Temps = %s" % (end - start))

out = sortieGlobal.copy_to_host()





outGrad = cuda.to_device(np.ascontiguousarray(np.random.randint(2, size=out.shape))) # ENTREE 3

inpGrad = cuda.device_array(IMG.shape)          # SORTIE 1
filtreGrad = cuda.device_array(filtre.shape)    # SORTIE 2


@cuda.jit
def convBack(I,F,Og, Ig, Fg):
    tx,ty,tz = cuda.threadIdx.x,cuda.threadIdx.y,cuda.threadIdx.z
    tblockx, tblocky,tblockz = cuda.blockIdx.x,cuda.blockIdx.y,cuda.blockIdx.z
    
    x,y,z=tx+(tblockx%(cuda.gridDim.x//3)*cuda.blockDim.x),ty+(tblocky%(cuda.gridDim.y//3)*cuda.blockDim.y),tz+tblockz*cuda.blockDim.z

    
    i,j = tblockx//(cuda.gridDim.x//3),tblocky//(cuda.gridDim.y//3)

    
    if(x<Og.shape[0] and y<Og.shape[1] and z<Og.shape[2]):
        
        #Ig[x+i,y+j,z] += I[x+i,y+j,z//F.shape[2]]*F[i,j,z%F.shape[2]]
        Fg[i,j,(z%Fg.shape[2])] += Og[x,y,z]*I[x+i,y+j, z//Fg.shape[2]]



threadParBlock = (16,16,4)
blocksParGrille = (3*int(ceil(out.shape[0]/16)),3*int(ceil(out.shape[1]/16)), int(ceil(out.shape[2]/4)))

start = t.time()
convBack[blocksParGrille, threadParBlock](IMGGlobal,filtreGlobal,outGrad, inpGrad, filtreGrad)
end = t.time()
print("Temps = %s" % (end - start))

test = filtreGrad.copy_to_host()
