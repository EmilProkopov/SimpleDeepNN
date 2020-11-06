# SimpleDeepNN
Realization of a deep neural network with an example of usage
The network is fully customizable: it's possible to set any amount of layers, number of neurons in each layer, and their activation functions.

`utils.py` contains realizations of activation and cost functions and their derivatives for forward and backward propagation (currently there are: ReLU, sigmoid and cross-entropy).

`neural_net.py` contains functions for training and prediction.

In `example_of_usage.py` the above is used for training a model that recognizes digits from Keras MNIST digits classification dataset. After the model is trained its accuracy is checked and parameters are saved to a .npy file so they can be used for prediction later.

Example of a learning curve:

`nice_params.py` is an example of parameters of a trained model. It contains params of a trained model with two hidden layers:
The input layer contains 784 neurons.
The first hidden layer contains 56 neurons with ReLU activation function. The second one contains 28 with the same activation. The output layer has ten neurons with sigmoid activation.
Model's accuracy on the training set: 98.82%;
On the test set: 93.51%
