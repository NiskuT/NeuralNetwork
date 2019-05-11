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
        r= data-wanted
        a=r*r
        b=0.5*a
        return b

    def backprop(self, data, wanted):

        miss = [self.cost(data[-1], wanted)]

        for k in range(1,len(self.weights)+1):
            
            der=data[-k]*(1.0-data[-k])


            chgt=miss[-1]*der

            deltaBias = self.app*np.dot
            delta= self.app * np.dot(chgt, data[-k-1].T)

            self.weights[-k][0] += delta
            print("Modif sur le poids:", len(self.weights)-k)
            miss.append(np.dot(self.weights[-k][0].T, miss[-1]))

        return 0


      
#if __name__ == "__main__":
    
N = NeuralNetwork(3,4,4,2)

a = N.forward(np.array([[2],[3],[4]]))
print(a)
for k in N.weights:
    print("\n\n")
    print(k[0])

print("C partie!")

N.backprop(a, np.array([[0.3],[0.5]]))
print("Done")

for k in N.weights:
    print("\n\n")
    print(k[0])
