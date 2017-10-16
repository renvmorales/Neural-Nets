# Neural-Nets
This repository assembles a set of coding and some other experiments using artificial neural networks (ANN) models.

Filenames on the format "ann#" provide coding examples for training different ANN models (backpropagation and Gradient Descent is implemented 'from scratch'). For these files, the '#' symbol denotes the considered number of hidden layers (1 or 2). Addtionaly the filename provides the type of activation function used (sigmoid, tanh, or relu) and whether network is designed for Multiclassification tasks. In all cases, synthetic training data is considered and randomly generated over the 2D space.
For example: 'ann2_sigmoid.py' implements a network with 2 hidden layers using the sigmoid as activation function for a binary classification task.


Upper case filenames provide object-oriented codes for fitting (training) and predicting ANN models. The code is general so it can automatically design a network with any number of hidden layer and hidden unities. For instance, the following trains and computing the prediction for a 4-layer ANN (considering each layer of 10 unities) using the sigmoid as activation:

model = ANN_sigmoid([10,10,10,10])

model.fit(X, Y, alpha=1e-4, epochs=10000, reg=0.01, show_fig=True)

Ypred = model.predict(X)
