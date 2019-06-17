#!/usr/bin/env python3

import numpy as np
import NNV2 as nn
import matplotlib.pyplot as plt
import random as rd

def toint(bytez):
    return int.from_bytes(bytez, byteorder='big')

def maxIndex(array):
    x,y=0,0
    tab = array.tolist()
    for k in range(len(tab)):
        if tab[k][0]>y:
            y=tab[k][0]
            x=k
    return x,y

def readImgData(ifile,lfile):
    imgz = open(ifile,'rb')
    lblz = open(lfile,'rb')
    
    imagic = toint(imgz.read(4))
    icount = toint(imgz.read(4))
    irows  = toint(imgz.read(4))
    icols  = toint(imgz.read(4))
    lmagic = toint(lblz.read(4))
    lcount = toint(lblz.read(4))
    
    # Checks
    if icount != lcount:
        raise ValueError("Les fichiers n'ont pas autant de données dans l'un que dans l'autre")
    if(imagic != 0x00000803):
        raise ValueError("Le fichier image n'est pas du bon type")
    if(lmagic != 0x00000801):
        raise ValueError("Le fichier labels n'est pas du bon type")
    # Data getting
    out = list()
    for ii in range(icount):
        #img = np.reshape(np.frombuffer(imgz.read(icols*irows),dtype=np.uint8),(icols,irows))
        img = np.frombuffer(imgz.read(icols*irows),dtype=np.uint8)

        lbl = toint(lblz.read(1))
        out.append([np.reshape(img, (784,1)),lbl])
        #print(ii)
    
    imgz.close()
    lblz.close()
    
    return out

mesImages = readImgData('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
'''
print(np.shape(mesImages[4][0]))
plt.imshow(np.reshape(mesImages[4][0], (28,28)))
plt.show()
'''
network = nn.NeuralNetwork()
'''
network.initWeights(784,16,16,10)

u=0
for t in range(20):
    for data in mesImages:
        want = np.zeros((10,1))
        want[data[1]] = 1

        network.updateWeightsBasic(data[0], want)
        u+=1
        if u%1000==0:
            print(str((u/600000)*100)+"%")

network.saveParam()
'''
network.openParam()

print("Phase de test")

for k in range(1):
    a = rd.randint(10,50000)
    b=network.forward(mesImages[a][0])
    plt.figure()
    plt.imshow( np.reshape(mesImages[a][0], (28,28)))
    
    print(maxIndex(b))
    plt.show()
'''
mesTests = readImgData('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')

k,l=0,0
for data in mesTests:

    p = network.forward(data[0])
    if data[1] == maxIndex(p)[0]:
        k+=1
    l+=1

print("TAUX DE REUSSITE: ", (k/l)*100, "%")

'''
