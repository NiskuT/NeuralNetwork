import numpy as np

class NeuralNetwork:
	
	def __init__(self, *args):

		self.weights = []
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

		return data[-1]

	def cost(self, data, wanted):
		return (data-wanted)**2

	def backprop(self, data, wanted):
		return 0





test = NeuralNetwork(3,2)
for k in test.weights:
	print(k)

print(test.forward(    [   np.random.random((3, 1))]       )     )
