import numpy as np
import matplotlib.pyplot as plt
from Activation import sigmoid, sigmoid_backward, relu, relu_backward

from enum import Enum


class AcvtivationFunc(Enum):
    sigmoid = 1
    relu = 2


def initialize_parameters_deep(layer_dims):
    """
    layer_dims -- a list containing the number of nodes of each layer
    parameters -- a dictionary containing weight and bias of each layer
                    Wl -- (layer_dims[l], layer_dims[l-1])
                    bl -- (layer_dims[l], 1)
    """
    
    np.random.seed()
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters


def linear_forward(A, W, b):
    """
    A -- activations from previous layer or inputs -- (size of previous layer, the number of examples)
    W -- weights matrix -- (size of l, size of l-1)
    b -- bias vector -- (size of the l, 1)
    Z -- the input of the activation function -- (size of previous layer, the number of examples)
    cache -- a tuple containing "A", "W" and "b" for computing the backprop
    """

    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation, annealing):
    """
    A_prev -- activations from previous layer or input data -- (size of previous layer, number of examples)
    W -- weights matrix -- （size of l, size of l-1)
    b -- bias vector -- (size of l, 1)
    activation -- the activation to be used in this layer, either "sigmoid" or "relu"
    A -- the output of the activation function
    cache -- a tuple containing "linear_cache" and "activation_cache" for backprop
    """
    
    if activation == AcvtivationFunc.sigmoid:
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z, annealing)
    
    elif activation == AcvtivationFunc.relu:
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters, annealing):
    """
    X -- input data -- (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    AL -- activation value for the last layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], AcvtivationFunc.relu, annealing)
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    A_prev = A
    AL, cache = linear_activation_forward(A_prev, parameters['W' + str(L)], parameters['b' + str(L)], AcvtivationFunc.sigmoid, annealing)
    caches.append(cache)
    
    assert(AL.shape == (1, X.shape[1]))
            
    return AL, caches


def compute_cost(AL, Y):
    """
    AL -- probability vector corresponding to your label predictions -- (1, number of examples)
    Y -- true desired vector -- (1, number of examples)
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    # cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1-AL), 1-Y)) / m 
    cost = np.sum((Y - AL)**2) / Y.shape[1]
    
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


def linear_backward(dZ, cache):
    """
    dZ -- Gradient of the cost with respect to the linear output of current layer l
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == AcvtivationFunc.relu:
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == AcvtivationFunc.sigmoid:
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    grads -- A dictionary with the gradients
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]

    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients.
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, AcvtivationFunc.sigmoid)
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, AcvtivationFunc.relu)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    parameters -- a dictionary containing your parameters 
    grads -- a dictionary containing your gradients, output of L_model_backward
    parameters -- a dictionary containing your updated parameters 
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters




