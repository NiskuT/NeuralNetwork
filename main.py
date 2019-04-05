import numpy as np
import random as rd

class NeuralNetwork:
	
	def __init__(self, *args):

		self.weights = []
		self.app = 0.05
		for k in range(1,len(args)):
			self.weights.append([(2*np.random.random((args[k],args[(k-1)]))-1), (2*np.random.random((args[k],1))-1)])

	def sigmoid(self, x):
		return (1/(1+np.exp(-x)))

	def Dsigm(self, x):
		return x*(1-x)

	def forward(self, data):

		for w in self.weights:
			try:
				data.append(self.sigmoid(np.dot(w[0],data[-1]) + w[1]))
			except ValueError:
				print("Erreur sur les valeurs.")
		return data


	def cost(self, data, wanted):
		return (data-wanted)**2

	def backprop(self, data, wanted):
		
		a=self.cost(data[-1], wanted)
		delta = [a]
		deltaP = [self.cost(data[-1], wanted)]

		for i in range(1, len(data)):

			delta.append( self.Dsigm(data[-i-1])*np.dot(self.weights[-i][0].T, delta[-1]) )
			deltaP.append( np.dot(self.weights[-i][1], delta[-1].T) )

		delta.reverse()
		deltaP.reverse()
		
		for i in range(len(self.weights)):
			for j in range(len(self.weights[i][0])):
				for k in range(len(self.weights[i][0][j])):
					self.weights[i][0][j,k] += data[i][k] * self.app * delta[i+1][j]

		for i in range(len(self.weights)):
			for j in range(len(self.weights[i][1])):
				for k in range(len(self.weights[i][1][j])):
					self.weights[i][1][j,k] +=  self.app * delta[i+1][j]

		return a

	def train(self):
		a= int(rd.getrandbits(1))
		b= int(rd.getrandbits(1))
		inpuT = np.array([[a],[b]])
		output = self.forward([inpuT])
		wanted = np.array([[a != b], [a^b]])

		a = self.backprop(output, wanted)
		return a


	def loop(self, n, e = 50):
		i=0
		for k in range(n):
			i+=1
			if i > e:
				print(self.train(), "numero", k)
				i=0
			else:
				self.train()







first = NeuralNetwork(2,4,4,2)
first.loop(30000, 1000)
print("les test!")
print(first.forward([np.array([[1], [1]])])[-1])
print(first.forward([np.array([[0], [1]])])[-1])
print(first.forward([np.array([[1], [0]])])[-1])
print(first.forward([np.array([[0], [0]])])[-1])
#for k in test.weights:
#	print(k)

#print( "\n\n\n\n", test.backprop(test.forward([np.random.random((3, 1))]), np.array([[0.7], [0.]])) )

#http://www.anyflo.com/bret/cours/rn/rn5.htm#exemple
