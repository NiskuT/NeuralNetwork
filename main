import numpy as np

class NeuralNetwork:
	
	def __init__(self, *args):

		self.weights = []
		for k in range(1,len(args)):
			self.weights.append((2*np.random.random((args[k],args[(k-1)]))-1))

	def sigmoid(self, x):
		return (1/(1+np.exp(-x)))

	def Dsigm(self, x):
		return x*(1-x)

	def forward(self, data):

		for w in self.weights:

			data = np.append(data)

		return 0

test = NeuralNetwork(11,4,7)
for k in test.weights:
	print(k)
