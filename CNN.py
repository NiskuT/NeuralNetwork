# -*- coding: utf-8 -*-

#from numba import cuda
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

        self.schema = schema
        self.filters = []
        self.cachePooling = []
        self.cacheConv = []
        self.train =True
        self.learningRate=0.01
        
        x=0
        for k in range(len(schema)):
            if schema[k][0] == 'c':
                self.filters.append(np.random.randn(3, 3,int(schema[k][1])) / 9)
                schema[k] = 'c'+str(x)
                x+=1
                # regarder pk diviser par 9 
        
        
        
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

        
        if self.train: self.cachePooling.append(np.zeros(( (imageBank.shape[0]+dim-hpadding),(imageBank.shape[1]+dim-vpadding), imageBank.shape[2])))
        
        newBank = np.zeros(((imageBank.shape[0]+dim-hpadding)//dim,\
                            (imageBank.shape[1]+dim-vpadding)//dim, imageBank.shape[2]))

        for k in range(newBank.shape[2]):
            
            im = np.zeros(((imageBank.shape[0]+dim-hpadding),(imageBank.shape[1]+dim-vpadding)))
            
            firstLine, lastLine=  (dim-hpadding)//2,  im.shape[0]-((dim-hpadding)//2+hpadding%2)
            firstCol, lastCol  =  (dim-vpadding)//2,  im.shape[1]-((dim-vpadding)//2+vpadding%2)

            im[  firstLine:lastLine, firstCol:lastCol ] += imageBank[:,:,k]
            
            if self.train: self.cachePooling[-1][:,:,k]+=im
            
            for i in range(newBank.shape[0]):
                for j in range(newBank.shape[1]):
                    
                    if Type == "m":
                        newBank[i,j,k]=np.max(im[dim*i:dim*(i+1), dim*j:dim*(j+1)])
                        
                    elif Type == "a":
                        newBank[i,j,k]=np.mean(im[dim*i:dim*(i+1), dim*j:dim*(j+1)])
                        
                        
        if self.train: self.cachePooling.append((firstLine, lastLine, firstCol,lastCol,dim))  
        
        return newBank
    
    
    
    def poolingBackprop(self, outputGrad):
        
        #regarder np.amax
        
        firstLine, lastLine, firstCol,lastCol,dim = self.cachePooling.pop()
        inputP = self.cachePooling.pop()

        inputGrad = np.zeros(inputP.shape)

        
        for k in range(inputP.shape[2]):
            
            for i in range(inputP.shape[0]):
                
                for j in range(inputP.shape[1]):

                    
                    if inputP[i,j,k]==np.max(inputP[(i//dim)*dim:(i//dim+1)*dim, (j//dim)*dim:(j//dim + 1)*dim, k]):
                        
                        inputGrad[i,j,k]=outputGrad[i//dim,j//dim,k]
                        
        return inputGrad[firstLine:lastLine,   firstCol:lastCol,  : ]
        
    
    
    

    def convolution(self, imageBank, filterBankID):
        '''
            imageBank: une matrice avec comme profondeur le nombre d'images 2D
            à traiter. Pour la première convolution on traite indépendent R,G,B
            Attention: il s'agit obligatoirement d'une matrice 3D
            
            filterBankID: l'indice de la banque de filtre
        '''
        
        ### regarder flatten()
        if self.train:
            self.cacheConv.append(imageBank)
            self.cacheConv.append(filterBankID)
            self.cacheConv.append(imageBank)
            
        filterBank = self.filters[filterBankID]
        
        

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
                print(image, filtre)
                output[:,:,filterBank.shape[2]*image+filtre]=\
                self.convWithSingleFilter(imageBank[:,:,image], filterBank[:,:,filtre])
                
        return output
        

    def convWithSingleFilter(self, img, filter1):
        #import time as t
        #start = t.time()


        # création de l'image de sorie
        out = np.zeros(((img.shape[0]-filter1.shape[0]+1),(img.shape[1]-filter1.shape[1]+1)))
        
        for k in range(out.shape[0]):
            for i in range(out.shape[1]):
                out[k,i]+=np.sum(img[k:k+filter1.shape[0], i:i+filter1.shape[1]]*filter1)
                
                
                
        #end = t.time()
        #print("Elapsed = %s" % (end - start))
        return out
    
    
    
    
    
    def convBackprop(self, grad):
        
        filterBankID=self.cacheConv.pop()
        inp = self.cacheConv.pop()
        
        inpGrad = np.zeros(inp.shape)
        filterGrad = np.zeros(self.filters[filterBankID].shape)
        

        
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                for k in range(grad.shape[2]):
                    
                    filterGrad[:,:,k%filterGrad.shape[2]] += grad[i,j,k] * inp[i:i+filterGrad.shape[0],j:j+filterGrad.shape[1],k]
                    inpGrad[i:i+filterGrad.shape[0],j:j+filterGrad.shape[1],k//filterGrad.shape[2]] += grad[i,j,k]*self.filters[filterBankID][k%filterGrad.shape[2]]
                    
        self.filters[filterBankID] -= self.learningRate * filterGrad
        
        return inpGrad
    
    def convBackpropWithSingleFilter(self):
        pass
        

        
    
    
        
        
'''
T=A.flatten('F')
aff(T.reshape((3,3,3), order='F'))
'''
    
    
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
'''        
a = CNN()
im = plt.imread("plage.jpg")
filterL=[]
#filterL=[np.array([[-1,0,-1],[0,0,0],[-1,0,-1]])]
filterL.append(np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
#filterL.append(np.eye((3,3)))
filterL.append(np.array([[0,1,0],[1,-1,1],[0,1,0]]))

#filterL.append(np.array([[-1,0,1],[-1,0,1],[-1,0,1]]))
#filterL.append(np.ones((3,3)))
#filterL.append(np.array([[-1,-1,-1],[0,0,0],[1,1,1]]))
#filterL.append(np.array([[-1,0,0,0,-1],[0,1,1,1,0],[0,1,-1,1,0],[0,1,1,1,0],[-1,0,0,0,-1]]))

a.filters.append(np.transpose(np.array(filterL), axes=(1,2,0)))

L= a.convolution(im, 0)


for k in range(L.shape[2]):
    plt.figure()
    plt.imshow(L[:,:,k])


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
'''# -*- coding: utf-8 -*-

#from numba import cuda
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

        self.schema = schema
        self.filters = []
        self.cachePooling = []
        self.cacheConv = []
        self.train =True
        self.learningRate=0.01
        
        x=0
        for k in range(len(schema)):
            if schema[k][0] == 'c':
                self.filters.append(np.random.randn(3, 3,int(schema[k][1])) / 9)
                schema[k] = 'c'+str(x)
                x+=1
                # regarder pk diviser par 9 
        
        
        
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

        
        if self.train: self.cachePooling.append(np.zeros(( (imageBank.shape[0]+dim-hpadding),(imageBank.shape[1]+dim-vpadding), imageBank.shape[2])))
        
        newBank = np.zeros(((imageBank.shape[0]+dim-hpadding)//dim,\
                            (imageBank.shape[1]+dim-vpadding)//dim, imageBank.shape[2]))

        for k in range(newBank.shape[2]):
            
            im = np.zeros(((imageBank.shape[0]+dim-hpadding),(imageBank.shape[1]+dim-vpadding)))
            
            firstLine, lastLine=  (dim-hpadding)//2,  im.shape[0]-((dim-hpadding)//2+hpadding%2)
            firstCol, lastCol  =  (dim-vpadding)//2,  im.shape[1]-((dim-vpadding)//2+vpadding%2)

            im[  firstLine:lastLine, firstCol:lastCol ] += imageBank[:,:,k]
            
            if self.train: self.cachePooling[-1][:,:,k]+=im
            
            for i in range(newBank.shape[0]):
                for j in range(newBank.shape[1]):
                    
                    if Type == "m":
                        newBank[i,j,k]=np.max(im[dim*i:dim*(i+1), dim*j:dim*(j+1)])
                        
                    elif Type == "a":
                        newBank[i,j,k]=np.mean(im[dim*i:dim*(i+1), dim*j:dim*(j+1)])
                        
                        
        if self.train: self.cachePooling.append((firstLine, lastLine, firstCol,lastCol,dim))  
        
        return newBank
    
    
    
    def poolingBackprop(self, outputGrad):
        
        #regarder np.amax
        
        firstLine, lastLine, firstCol,lastCol,dim = self.cachePooling.pop()
        inputP = self.cachePooling.pop()

        inputGrad = np.zeros(inputP.shape)

        
        for k in range(inputP.shape[2]):
            
            for i in range(inputP.shape[0]):
                
                for j in range(inputP.shape[1]):

                    
                    if inputP[i,j,k]==np.max(inputP[(i//dim)*dim:(i//dim+1)*dim, (j//dim)*dim:(j//dim + 1)*dim, k]):
                        
                        inputGrad[i,j,k]=outputGrad[i//dim,j//dim,k]
                        
        return inputGrad[firstLine:lastLine,   firstCol:lastCol,  : ]
        
    
    
    

    def convolution(self, imageBank, filterBankID):
        '''
            imageBank: une matrice avec comme profondeur le nombre d'images 2D
            à traiter. Pour la première convolution on traite indépendent R,G,B
            Attention: il s'agit obligatoirement d'une matrice 3D
            
            filterBankID: l'indice de la banque de filtre
        '''
        
        ### regarder flatten()
        if self.train:
            self.cacheConv.append(imageBank)
            self.cacheConv.append(filterBankID)
            self.cacheConv.append(imageBank)
            
        filterBank = self.filters[filterBankID]
        
        

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
                print(image, filtre)
                output[:,:,filterBank.shape[2]*image+filtre]=\
                self.convWithSingleFilter(imageBank[:,:,image], filterBank[:,:,filtre])
                
        return output
        

    def convWithSingleFilter(self, img, filter1):
        #import time as t
        #start = t.time()


        # création de l'image de sorie
        out = np.zeros(((img.shape[0]-filter1.shape[0]+1),(img.shape[1]-filter1.shape[1]+1)))
        
        for k in range(out.shape[0]):
            for i in range(out.shape[1]):
                out[k,i]+=np.sum(img[k:k+filter1.shape[0], i:i+filter1.shape[1]]*filter1)
                
                
                
        #end = t.time()
        #print("Elapsed = %s" % (end - start))
        return out
    
    
    
    
    
    def convBackprop(self, grad):
        
        filterBankID=self.cacheConv.pop()
        inp = self.cacheConv.pop()
        
        inpGrad = np.zeros(inp.shape)
        filterGrad = np.zeros(self.filters[filterBankID].shape)
        

        
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                for k in range(grad.shape[2]):
                    
                    filterGrad[:,:,k%filterGrad.shape[2]] += grad[i,j,k] * inp[i:i+filterGrad.shape[0],j:j+filterGrad.shape[1],k]
                    inpGrad[i:i+filterGrad.shape[0],j:j+filterGrad.shape[1],k//filterGrad.shape[2]] += grad[i,j,k]*self.filters[filterBankID][k%filterGrad.shape[2]]
                    
        self.filters[filterBankID] -= self.learningRate * filterGrad
        
        return inpGrad
    
    def convBackpropWithSingleFilter(self):
        pass
        

        
    
    
        
        
'''
T=A.flatten('F')
aff(T.reshape((3,3,3), order='F'))
'''
    
    
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
'''        
a = CNN()
im = plt.imread("plage.jpg")
filterL=[]
#filterL=[np.array([[-1,0,-1],[0,0,0],[-1,0,-1]])]
filterL.append(np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
#filterL.append(np.eye((3,3)))
filterL.append(np.array([[0,1,0],[1,-1,1],[0,1,0]]))

#filterL.append(np.array([[-1,0,1],[-1,0,1],[-1,0,1]]))
#filterL.append(np.ones((3,3)))
#filterL.append(np.array([[-1,-1,-1],[0,0,0],[1,1,1]]))
#filterL.append(np.array([[-1,0,0,0,-1],[0,1,1,1,0],[0,1,-1,1,0],[0,1,1,1,0],[-1,0,0,0,-1]]))

a.filters.append(np.transpose(np.array(filterL), axes=(1,2,0)))

L= a.convolution(im, 0)


for k in range(L.shape[2]):
    plt.figure()
    plt.imshow(L[:,:,k])


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
