# -*- coding: utf-8 -*-


import numpy as np
import random as rd


class NeuralNetwork:
    
    def __init__(self, *args):
        """
            Entrer en paramètre le nombre de neuronnes par couche
            
            Le tableau weights contient deux colonnes:
            colonne 0: les matrices de poids entre deux couches, cad chaque matrice
            contient dans la colonne k les poids reliés aux neuronnes d'entrée k
            et la ligne l les poids reliés aux neuronnes d'arrivée l
                
            colonne 1: les biais en matrice colonne
            
            app: taux d'apprentissage du réseau
        """      
        self.numOfLayer = len(args)
        self.weights = []
        self.bias = []
        self.app = 0.005
        
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



      
#if __name__ == "__main__":
    
N = NeuralNetwork(2,4,1)

k=0
l=1000000
while k<l:
	N.train()
	k+=1
	if k%100==0:
		print((k/l)*100, "%")


print(N.forward(np.array([[40],[60]])))
print(N.forward(np.array([[60],[40]])))
print(N.forward(np.array([[35],[60]])))
print(N.forward(np.array([[15],[20]])))
'''
for k in N.weights:
    print("\n\n")
    print(k[0])'''
