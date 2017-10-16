# Example of 1-hidden layer neural network for multiclass classification 
# using a 'tanh' activation function
import numpy as np 
import matplotlib.pyplot as plt 
from ann_functions import *




# number of samples for each class
N_class = 400 


# generates random 2-D points 
X1 = np.random.randn(N_class,2)+np.array([2,2])
X2 = np.random.randn(N_class,2)+np.array([-2,-2])
X3 = np.random.randn(N_class,2)+np.array([-2,2])
X4 = np.random.randn(N_class,2)+np.array([2,-2])
X = np.vstack([X1, X2, X3, X4])

# labels associated to the input
Y = np.array([0]*N_class+[1]*N_class+[2]*N_class+[3]*N_class)
# Y = np.reshape(Y, (len(Y),1))


# scatter plot of original labeled data
plt.scatter(X[:,0],X[:,1],c=Y,s=50,alpha=0.5)
plt.show()





# defining the parameters of the 1-hidden layer neural net
N = len(Y) # number of samples
D = X.shape[1]  # number of features
M = 5 	# number of hidden units
K = 4  	# number of the output dimension



# generates an indicator target matrix
Trgt = np.zeros((N, K))
Trgt[np.arange(N), Y.astype(np.int32)] = 1




# initializing the hidden layer with random weights
W0, b0 = init_weights(D, M)
W1, b1 = init_weights(M, K)





# optimization settings
Nsteps = 10000  # number of iterations for gradient ascend
alpha = 0.0001  # learning rate
# lambda_reg = 0.1 # regularization parameter
J = np.zeros(Nsteps)  # array for the cost values at each step





# Gradient ascend loop - the 'cost function' used here is a concave 
# function so the algorithm is finding local maxima rather than the 
# local minima
for i in range(Nsteps):	
	Z1 = forward_step_tanh(X, W0, b0)
	PY = forward_step(Z1, W1, b1) # output values are in the range [0,1]
	PY = softmax(PY)
	J[i] = cross_entropy_multi(Trgt, PY)
	W1buf = W1
	W1 += alpha * Z1.T.dot(Trgt-PY)
	b1 += alpha * (Trgt-PY).sum(axis=0)
	W0 += alpha * X.T.dot((Trgt-PY).dot(W1buf.T) * (1-Z1**2))
	b0 += alpha * ((Trgt-PY).dot(W1buf.T) * (1-Z1**2)).sum(axis=0)
	Pred = np.argmax(PY, axis=1)
	if i % 100 == 0:
		print('Epoch:',i,' Cost:{:.4f}'.format(J[i]), 
			" Accuracy:{:1.4f}".format(np.mean(Y==Pred)))



# plots the evolution of the cost function value through the optimization
plt.plot(np.arange(Nsteps), J)
plt.show()
