# -*- coding: utf-8 -*-

import numpy as np
import pickle as pk

class NeuralNetwork:

    def __init__(self):
        """
            app: taux d'apprentissage du réseau
        """      

        self.app = 0.005

    def initWeights(self, *args):
        '''
            Entrer en paramètre le nombre de neuronnes par couche
            Le tableau weights contient deux colonnes:
            colonne 0: les matrices de poids entre deux couches, cad chaque matrice
            contient dans la colonne k les poids reliés aux neuronnes d'entrée k
            et la ligne l les poids reliés aux neuronnes d'arrivée l
            colonne 1: les biais en matrice colonne
        '''

        self.weights = []
        self.bias = []
        self.numOfLayer = len(args)   
        self.data = 0

        for k in range(1,len(args)):            

            self.weights.append( (2*np.random.random((args[k],args[(k-1)]))-1) )
            self.bias.append(   (2*np.random.random((args[k],1))-1)   )


    def sigmoid(self, x):
        return (1/(1+np.exp(-x)))


    def Dsigm(self, x):
        ''' Dérivée de la sigmoid '''
        return x*(1.0-x)

    def cost(self, data, wanted):
        """ 
        Fonction de cout
        squared error function
        """
        return (data - wanted)

    def maxIndex(self, array):
        x,y=0,0
        tab = array.tolist()
        for k in range(len(tab)):
            if tab[k][0]>y:
                y=tab[k][0]
                x=k
        return x,y


    def forward(self, inp, training=False):
        """
            Fonction principal du réseau:
            Prend en paramètres un ndarray colonne de meme taille que la couche 1
            Renvoie un tableau contenant la valeur de chaque neuronne
            Utiliser data[-1] pour obtenir la sortie final        
        """

        data = [inp]

        for w,b in zip(self.weights, self.bias):
            try:
                data.append( self.sigmoid(np.dot(w,data[-1]) + b) )
            except ValueError:
                print("Erreur sur les valeurs.")

        if training:
            self.data = data
        else:
            return data[-1]

    def backprop(self, wanted):
        

        deltaW = [np.zeros(w.shape) for w in self.weights]
        deltaB = [np.zeros(b.shape) for b in self.bias] 

        delta = self.cost(self.data[-1], wanted) * self.Dsigm(self.data[-1])

        deltaB[-1] = delta
        deltaW[-1] = np.dot(delta, self.data[-2].T)


        for k in range(2, self.numOfLayer):

            delta = np.dot(self.weights[-k+1].T, delta) * self.Dsigm(self.data[-k])

            deltaB[-k] = delta

            deltaW[-k] = np.dot(delta, self.data[-k-1].T)


        inpGrad = np.dot(self.weights[0].T, delta) * self.Dsigm(self.data[0])
        
        for w in range(self.numOfLayer-1):
            self.weights[w] -= self.app*deltaW[w]

        for b in range(self.numOfLayer-1):
            self.bias[b] -= self.app*deltaB[b]
        
        return inpGrad


    def saveParam(self):

        '''
        Sauvergarde les poids et biais en .txt
        '''

        with open("weights.txt", "wb") as fp:
            pk.dump(self.weights, fp)

        with open("bias.txt", "wb") as fp:
            pk.dump(self.bias, fp)

    def openParam(self):
        '''
        Ouvre des poids et biais enregistrés dans un fichier .txt
        '''
        with open("weights.txt", "rb") as fp:
            self.weights = pk.load(fp)

        with open("bias.txt", "rb") as fp:
            self.bias = pk.load(fp)

    def showW(self):
        print("Les poids:\n")
        for w in self.weights:
            print(w,'\n')

        print("Les biais:\n")
        for b in self.bias:
            print(b,'\n')

"""
N = NeuralNetwork()
N.initWeights(2,4,1)
print(N.forward(np.array([[40],[60]]),  True))
"""

