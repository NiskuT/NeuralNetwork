# -*- coding: utf-8 -*-


import numpy as np
import random as rd
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

    def updateWeightsBasic(self, inp, wanted):

        deltaW, deltaB = self.backprop(inp, wanted)
        #self.weights = [self.app*(w-delta) for w, delta in zip(self.weights, deltaW)]
        #self.bias = [self.app*(b-delta) for b, delta in zip(self.bias, deltaB)]
        for w in range(self.numOfLayer-1):
            self.weights[w] -= self.app*deltaW[w]

        for b in range(self.numOfLayer-1):
            self.bias[b] -= self.app*deltaB[b]





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
            return data
        else:
            return data[-1]



    def backprop(self, data, wanted):

        deltaW = [np.zeros(w.shape) for w in self.weights]
        deltaB = [np.zeros(b.shape) for b in self.bias] 


        activity = self.forward(data, True)
        delta = self.cost(activity[-1], wanted) * self.Dsigm(activity[-1])

        deltaB[-1] = delta
        deltaW[-1] = np.dot(delta, activity[-2].T)

        for k in range(2, self.numOfLayer):
            delta = np.dot(self.weights[-k+1].T, delta) * self.Dsigm(activity[-k])
            deltaB[-k] = delta
            deltaW[-k] = np.dot(delta, activity[-k-1].T)

        return (deltaW, deltaB)

    def train(self):
        a= rd.randrange(10,100)
        b= rd.randrange(10,100)
        inpuT = np.array([[a],[b]])
        wanted = np.array([[a<b]])

        self.updateWeightsBasic(inpuT, wanted)

    def imgTrain(self , training_data , epochs, test_data=None):

        if test_data:
            n_test = len(test_data)
            n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
        for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch , eta)

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

'''
    
N = NeuralNetwork()
N.openParam()
N.showW()

N.initWeights(2,4,1)

print(N.forward(np.array([[40],[60]])))
print(N.forward(np.array([[60],[40]])))
print(N.forward(np.array([[35],[60]])))
print(N.forward(np.array([[15],[20]])))

'''
