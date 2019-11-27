# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt



class CNN:
    
    def __init__(self, schema=[] ):
        '''
            Prend en paramètre le schema du CNN sous forme de liste:
            Pour une etape de convultion:
                "c[nbDeFiltre]" ex: "c8" pour une convultion avec 8 filtres
            Pour une etape de pooling:
                "pm[dim]" pour un max pooling ex: "pm2" pour un pooling "2x2"
                "pa[dim]" pour un average pooling : "pa3" pour un pooling "3x3"
            Pour une etape de non-linéarisation (qui evite notamment le 
            sur-apprentissage):
                "r" pour relu
                "s" pour sigmoid
                            
        '''
        num_filters = 1
        self.schema = schema
        self.filters = np.random.randn(3, 3,num_filters) / 9
        self.cachePooling = []
        self.train =True

        
        
        
    def sigmoid(self,x):
        return (1/(1+np.exp(x)))
    
    def relu(self,x):
        return max(0,x)
    
    def pooling(self, Type, imageBank, dim):
        """
            Fonction de pooling pour image 2D (nuance de gris)
            type: m pour max pooling 
                  a pour average pooling
            imageBank est une image de profondeur le nombre d'images 
            = une liste de matrices 2D
        """
        hpadding = imageBank.shape[0]%dim
        vpadding = imageBank.shape[1]%dim

        
        if self.train: self.cachePooling.append([])
        
        newBank = np.zeros(((imageBank.shape[0]+dim-hpadding)//dim,\
                            (imageBank.shape[1]+dim-vpadding)//dim, imageBank.shape[2]))

        for k in range(newBank.shape[2]):
            
            im = np.zeros(((imageBank.shape[0]+dim-hpadding),(imageBank.shape[1]+dim-vpadding)))
            
            firstLine, lastLine=  (dim-hpadding)//2,  im.shape[0]-(hpadding//2+hpadding%2)
            firstCol, lastCol  =  (dim-vpadding)//2,  im.shape[1]-(vpadding//2+vpadding%2)
            
            im[  firstLine:lastLine, firstCol:lastCol ] += imageBank[:,:,k]
            
            if self.train: self.cachePooling[-1].append(im)
            
            for i in range(newBank.shape[0]):
                for j in range(newBank.shape[1]):
                    
                    if Type == "m":
                        newBank[i,j,k]=np.max(im[dim*i:dim*(i+1), dim*j:dim*(j+1)])
                        
                    elif Type == "a":
                        newBank[i,j,k]=np.mean(im[dim*i:dim*(i+1), dim*j:dim*(j+1)])
                        
                        
        if self.train: self.cachePooling[-1].append((firstLine, lastLine, firstCol,lastCol,dim))  
        
        return newBank
    
    
    
    def poolingBackprop(self, outputGrad):
        
        #regarder np.amax
        
        firstLine, lastLine, firstCol,lastCol,dim = self.cachePooling[-1].pop()
        inputP = self.cachePooling.pop()
        
        inputGrad = np.zeros((inputP[0].shape[0],inputP[0].shape[1], len(inputP)))

        
        for k in range(len(inputP)):
            
            for i in range(inputP[k].shape[0]):
                
                for j in range(inputP[k].shape[1]):

                    
                    if inputP[k][i,j]==np.max(inputP[k][(i//dim)*dim:(i//dim+1)*dim, (j//dim)*dim:(j//dim + 1)*dim]):
                        
                        inputGrad[i,j,k]=outputGrad[i//dim,j//dim,k]
                        
        return inputGrad[firstLine:lastLine,   firstCol:lastCol,  : ]
        
    
    
    
    
    def convolution(self, imageBank, filterBank):
        '''
            imageBank: une matrice avec comme profondeur le nombre d'images 2D
            à traiter. Pour la première convolution on traite indépendent R,G,B
            Attention: il s'agit obligatoirement d'une matrice 3D
            
            filterBank: une matrice de taille n,n,y avec n la dimension des filtres
            et y le nombre de filtres
        '''
        
        ### regarder flatten()

        # verification des filtres
        if filterBank.shape[0] != filterBank.shape[1] or \
        filterBank.shape[0] > imageBank.shape[0] or \
        filterBank.shape[0]%2==0 or len(filterBank.shape)!=3:
            # On attend des filtres qu'ils soient de dimension inférieur à celle
            # de l'image, carrés, et de dimension impaire (pour avoir un pixel centré)
            # On attend également que filterBank soit une matrice 3D
            raise NameError("Erreur_filtre")
            
            
            
        output = np.zeros((imageBank.shape[0]-filterBank.shape[0]+1, \
                           imageBank.shape[1]-filterBank.shape[1]+1, \
                           imageBank.shape[2] * filterBank.shape[2]  ))
        
        for image in range( imageBank.shape[2] ):
            for filtre in range( filterBank.shape[2] ):
                
                output.append(self.convWithSingleFilter(imageBank[:,:,image], filterBank[:,:,filtre]))
                
                
                
                
            
        #return np.transpose(np.array(output), axes=(1,2,0))
        return output
        
    
    def convWithSingleFilter(self, img, filter1):
        
        # création de l'image de sorie
        out = np.zeros(((img.shape[0]-filter1.shape[0]+1),(img.shape[1]-filter1.shape[1]+1)))
        
        for k in range(out.shape[0]):
            for i in range(out.shape[1]):
                out[k,i]+=np.sum(img[k:k+filter1.shape[0], i:i+filter1.shape[1]]*filter1)

        return out
    
    
def aff(a):
    for j in range(a.shape[2]):
        for k in range(a.shape[0]):
            for i in range(a.shape[1]):
                print(int(a[k,i,j]), end='   ')
            print()
        print("\n")
        
def aff1(a):
    for k in range(a.shape[0]):
        for i in range(a.shape[1]):
            print(int(a[k,i]), end='   ')
        print()
        
a = CNN()
im = plt.imread("plage.jpg")
filterL=[]
#filterL=[np.array([[-1,0,-1],[0,0,0],[-1,0,-1]])]
#filterL.append(np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
#filterL.append(np.eye((3,3)))
filterL.append(np.array([[0,1,0],[1,-1,1],[0,1,0]]))

#filterL.append(np.array([[-1,0,1],[-1,0,1],[-1,0,1]]))
#filterL.append(np.ones((3,3)))
#filterL.append(np.array([[-1,-1,-1],[0,0,0],[1,1,1]]))
#filterL.append(np.array([[-1,0,0,0,-1],[0,1,1,1,0],[0,1,-1,1,0],[0,1,1,1,0],[-1,0,0,0,-1]]))

L= a.convolution(im, np.transpose(np.array(filterL), axes=(1,2,0)))


for k in range(L.shape[2]):
    plt.figure()
    plt.imshow(L[:,:,k])


'''
def filtre(k):
    if k ==0:
        return a.convOfSingleFilter(im, filterL[k])
    print("filtre ",k)
    return a.convOfSingleFilter(filtre(k-1), filterL[k])
plt.imshow(a.pooling("m", np.array(im) , 7)[:,:,1])
plt.figure()
plt.imshow(a.pooling("m", np.array(im) , 7)[:,:,0])
plt.figure()
plt.imshow(a.pooling("m", np.array(im) , 7)[:,:,2])
#plt.imshow(a.convOfSingleFilter(im, filterL[6]))
'''
