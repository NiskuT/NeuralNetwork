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
        
        self.weights = []
        self.app = 0.05
        
        for k in range(1,len(args)):
            
            self.weights.append([(2*np.random.random((args[k],args[(k-1)]))-1), (2*np.random.random((args[k],1))-1)])            



    def sigmoid(self, x):
        return (1/(1+np.exp(-x)))


    def forward(self, inp):
        """
            Fonction principal du réseau:
            Prend en paramètres un ndarray colonne de meme taille que la couche 1
            
            Renvoie un tableau contenant la valeur de chaque neuronne
            Utiliser data[-1] pour obtenir la sortie final
        
        """
        data = [inp]
        for w in self.weights:
            try:
                data.append(self.sigmoid(np.dot(w[0],data[-1]) + w[1]))
            except ValueError:
                print("Erreur sur les valeurs.")
                
        return data

    def cost(self, data, wanted):
        """ 
        Fonction de cout
        squared error function
        """
        return (1/2)*(data-wanted)**2

    def backprop(self, data, wanted):
        pass


      
if __name__ == "__main__":
    
    N = NeuralNetwork(3,4,2)

    print(N.forward(np.array([[2],[3],[4]]))[-1])




















        
