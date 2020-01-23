# -*- coding: utf-8 -*-
from CNN import CNN
from NN import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

conv = CNN(['c2','pm2', 'c2','pm2', 'c2'])
neuralNet = NeuralNetwork()
k=584
neuralNet.initWeights(k, k//2,k//2,20,1)

def forward(image):
    conv.train = False
    last = image
    for k in conv.schema:
        
        if k[0]=='c':
            last = conv.convolution(last, int(k[1]))
            
        if k[0]=='p':
            last = conv.pooling(k[1], last, int(k[2]))
            
    last=last.flatten('F')
    return neuralNet.forward(last)




def forwardWithTrain(image, wanted):
    conv.train= True
    last = image
    for k in conv.schema:
        
        if k[0]=='c':
            last = conv.convolution(last, int(k[1]))
            
        if k[0]=='p':
            last = conv.pooling(k[1], last, int(k[2]))
            
    dim=last.shape
    last=last.flatten('F')
    neuralNet.forward(last, training=True)
    
    grad = neuralNet.backprop(wanted)
    
    grad=grad.reshape(dim, order='F')
    
    
    for k in range(len(conv.schema)):
        
        if conv.schema[-k][0]=='c':
            grad = conv.convBackprop(grad)
            
        if conv.schema[-k][0]=='p':
            grad = conv.poolingBackprop(grad)
            
    return 0

