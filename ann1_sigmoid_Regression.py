# Example of 1-hidden layer neural network for regression 
# using a logistic activation function
import numpy as np 
import matplotlib.pyplot as plt 
from ann_functions import *




N = 200
X = np.reshape(np.linspace(-10,10,N), (N,1))
Y = X**2 + 2*X + 8 + 2*np.random.randn(N,1)


plt.plot(X[:,0],Y,'.-')
plt.show()




# defining the parameters of the 1-hidden layer neural net
# N = len(Y) # number of samples
D = X.shape[1]  # number of features
M = 20 	# number of hidden units
K = 1  	# number of the output dimension




# initializing the hidden layer with random weights
W0, b0 = init_weights(D, M)
W1, b1 = init_weights(M, K)





# optimization settings
Nsteps = 50000  # number of iterations for gradient descend
alpha = 0.00001  # learning rate
# lambda_reg = 0.1 # regularization parameter
J = np.zeros(Nsteps)  # array for the cost values at each step





# Gradient descend loop - the 'cost function' used here is a convex 
# function so the algorithm is finding local minima
for i in range(Nsteps):	
	Z = forward_step_sigmoid(X, W0, b0)
	Ypred = forward_step(Z, W1, b1) # output values are in the range [0,1]
	J[i] = resid_squares(Y, Ypred)
	W1buf = W1
	W1 -= alpha * Z.T.dot(Ypred-Y)
	b1 -= alpha * (Ypred-Y).sum(axis=0)
	W0 -= alpha * X.T.dot((Ypred-Y).dot(W1buf.T) * Z * (1-Z))
	b0 -= alpha * ((Ypred-Y).dot(W1buf.T) * Z * (1-Z)).sum(axis=0)
	if i % 100 == 0:
		print('Epoch:',i,' Cost:{:.4f}'.format(J[i]), 
			' Rmse:{:.4f}'.format(np.sqrt(np.mean((Y-Ypred)**2))) )




# plots the evolution of the cost function value through the optimization
plt.plot(np.arange(Nsteps), J, label='Cost function J')
plt.legend()
plt.show()



# plots on the same graph original data and predicted values by 
# the 1-layer ANN model
plt.plot(X, Y, '.-', label='Original data')
plt.plot(X, Ypred, 'r-', label='Predicted values')
plt.legend()
plt.show()
