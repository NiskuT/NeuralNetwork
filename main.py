import numpy as np

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
				print("Erreur sur les valeurs d'entrée.")

		return data

	def cost(self, data, wanted):
		return (data-wanted)**2

	def backprop(self, data, wanted):
		
		delta = [self.cost(data[-1], wanted)]

		for i in range(1, len(data)):

			delta.append( self.Dsigm(data[-i-1])*np.dot(self.weights[-i][0].T, delta[-1]) )

		delta.reverse()

		print(self.weights, "\n\n")
		print(delta, "\n\n")

		
		for i in range(len(self.weights)):
			for j in range(len(self.weights[i][0])):
				for k in range(len(self.weights[i][0][j])):
					self.weights[i][0][j,k] *= data[i][k] * self.app * delta[i+1][j]

		print(self.weights)

		return 0





test = NeuralNetwork(3,4,2)
#for k in test.weights:
#	print(k)

print( "\n\n\n\n", test.backprop(test.forward([np.random.random((3, 1))]), np.array([[0.7], [0.]]))     )
