import numpy as np
from enum import Enum


def sigmoid(Z, annealing):
    """
    Z --- numpy array
    A --- output of sigmoid function which has the same shape as Z
    cache ---  storage for backpropagation
    """

    A = 1/(1+np.exp(-Z / annealing))
    
    cache = Z
    
    return A, cache



def relu(Z):
    """
    Z -- Output of the linear layer, of any shape
    A -- has the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backprop
    """
    
    A = np.maximum(0,Z)   
    # assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <=  set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    # assert (dZ.shape == Z.shape)
    
    return dZ


def sigmoid_backward(dA, cache):
    """
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    # assert (dZ.shape == Z.shape)
    
    return dZ

