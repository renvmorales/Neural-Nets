# This API implements the following tasks:
#  - Multi-layer 'sigmoid' ANN models for binary classification problems
#  - Standard GD optimization 
# 
# Implementation from scratch (mainly using Numpy).
import numpy as np
import matplotlib.pyplot as plt
from  ann_functions import *
import time



class ANN_sigmoid(object):
	def __init__(self, M):
		# this assures that all hidden unities are stored in a list
		if isinstance(M, int):
			self.M = [M]  # in case there is a single hidden layer...
		else:
			self.M = M



	def fit(self, X, Y, alpha=1e-3, reg=1e-4, epochs=5000, show_fig=False):
		N, D = X.shape
		K = 1

		self.N = N  # this variable will be used for normalization
		self.D = D  # store the dimension of the training dataset
		self.K = K  # output dimension
		
		# stores all hyperparameter values
		self.hyperparameters = {'alpha':alpha, 'reg':reg, 'epochs':epochs}


		# creates a list with the number of hidden unities (+ input/output)
		hdn_unties = [D] + self.M + [K]
		self.W = []
		self.b = []
		# initializes all weights randomly
		for k in range(1,len(self.M)+2):
			W, b = init_weights(hdn_unties[k-1], hdn_unties[k])
			self.W.append(W)
			self.b.append(b)


		J = np.zeros(epochs) # this array stores the cost with respect to each epoch
		start = time.time() # <-- starts measuring the optimization time from this point on...

		for i in range(epochs):  # optimization loop
			PY = self.forward(X)
			J[i] = cross_entropy_bin(Y, PY)
			self.back_prop(Y, PY, alpha, reg)
			if i % 100 == 0:
				print('Epoch:',i,' Cost: {:.4f}'.format(J[i]), 
					" Accuracy: {:1.4f}".format(np.mean(Y==np.around(PY))))


		end = time.time()
		self.elapsed_t = (end-start)/60 # total elapsed time
		self.cost = J # stores all cost values 

		print('\nOptimization complete')
		print('\nElapsed time: {:.3f} min'.format(self.elapsed_t))


		# customized plot with the resulting cost values
		if show_fig:
			plt.plot(J, label='Cost function J')
			plt.title('Evolution of the Cost through a GD optimization     Total runtime: {:.3f} min'.format(self.elapsed_t)+'    Final Accuracy: {:.3f}'.format(np.mean(Y==self.predict(X))))
			plt.xlabel('Epochs')
			plt.ylabel('Cost')
			plt.legend()
			plt.show()


	

	def forward(self, X):
		self.Z = [X] # this list contains all hidden unities + input/output
		for i in range(0,len(self.M)+1):
			self.Z.append(sigmoid(self.Z[i].dot(self.W[i]) + self.b[i]))
		return self.Z[-1]



	def back_prop(self, Y, PY, alpha, reg):
		N = len(Y)
		dZ = (PY-Y)/N
		Z = self.Z[:-1]
		Wbuf = self.W
		for i in range(1,len(self.W)+1):
			self.W[-i] -= alpha * (Z[-i].T.dot(dZ) + reg/(2*N)*self.W[-i])
			self.b[-i] -= alpha * (dZ.sum(axis=0) + reg/(2*N)*self.b[-i])
			dZ = dZ.dot(Wbuf[-i].T) * Z[-i]*(1-Z[-i])



	def predict(self, X):
		PY = self.forward(X)
		return np.around(PY)




def main():
# number of samples for each binary class
	N_class = 1000 


# generate random 2-D points 
	X1 = np.random.randn(N_class,2)+np.array([1,1])
	X2 = np.random.randn(N_class,2)+np.array([-1,-1])
	X = np.vstack([X1,X2])


# labels associated to the input
	Y = np.array([0]*N_class+[1]*N_class)
	Y = np.reshape(Y, (len(Y),1))


# general data information for the training process
	print('Total input samples:',X.shape[0])
	print('Data dimension:',X.shape[1])
	print('Number of output classes:',len(np.unique(Y)))
	print('\n')


# scatter plot of original labeled data
	plt.scatter(X[:,0],X[:,1],c=Y,s=50,alpha=0.5)
	plt.show()


# create an ANN model with the specified 4 hidden layers
	model = ANN_sigmoid([5,5,5,5])


# fits the model with the hyperparameters set
	model.fit(X, Y, alpha=1e-1, epochs=5000, reg=0, show_fig=True)
	

# compute the model accuracy
	Ypred = model.predict(X)
	print('\nFinal model accuracy: {:.4f}'.format(np.mean(Y==Ypred)))




if __name__ == '__main__':
    main()