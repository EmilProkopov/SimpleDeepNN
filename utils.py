import numpy as np

safe_c = 0.0001

def sigmoid(x):
    """
    Implementation of the sigmoid function
    
    Parameters
    ----------
    x : number, array or numpy array

    Returns
    -------
    number or numpy array
    """

    A = 1/(1+np.exp(-x))
    cache = x
    
    return A, cache


def relu(x):
    """
    Implementation of the relu function
    
    Parameters
    ----------
    x : number, array or numpy array

    Returns
    -------
    number or numpy array
    """

    A = np.maximum(0, x)
    
    cache = x 
    return A, cache


def relu_backward(dA, cache):
    """
    Backward propagation for a single RELU unit.
    
    Parameters
    ----------
    dA : numpy array
        post-activation gradient
    cache : numpy array
        Items stored for computing backward propagation efficiently
    
    Returns
    ----------
    dZ : numpy array
        gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    return dZ


def sigmoid_backward(dA, cache):
    """
    Backward propagation for a single SIGMOID unit.
    
    Parameters
    ----------
    dA : numpy array
        post-activation gradient
    cache : numpy array
        Items stored for computing backward propagation efficiently
    
    Returns
    ----------
    dZ : numpy array
        gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ


def cost_cross_entropy(A, Y):
    """
    The cross-entropy cost function.

    Parameters
    ----------
    A : numpy array
        probability vector corresponding to label predictions of shape:
        (1, number of examples)
    Y : numpy
        array true "label" vector of shape: (x, number of examples)

    Returns
    ----------
    cost : number
        cross-entropy cost
    """
    m = Y.shape[1]

    cost = - np.sum(Y * np.log(A + safe_c) + (1-Y) * np.log(1-A + safe_c)) / m    
    cost = np.squeeze(cost)      # turns [[n]] into n.
    
    return cost


def cost_cross_entropy_backward(A, Y):
    """
    Parameters
    ----------
    A : numpy array
        probability vector, output of the forward propagation
    Y : numpy array
        true lables

    Returns
    ----------
    derivative of the cross-entrophy cost function
    """
    return - (np.divide(Y, A+safe_c) - np.divide(1 - Y, 1 - A+safe_c))


def cost_MSE(A, Y):
    """
    The MSE cost function.

    Parameters
    ----------
    A : numpy array
        probability vector corresponding to label predictions of shape:
        (1, number of examples)
    Y : numpy
        array true "label" vector of shape: (x, number of examples)

    Returns
    ----------
    cost : number
        MSE cost
    """
    m = Y.shape[1]
    return np.sum((A - Y)**2) / m


def cost_MSE_backward(A, Y):
    """
    Parameters
    ----------
    A : numpy array
        probability vector, output of the forward propagation
    Y : numpy array
        true lables

    Returns
    ----------
    derivative of the MSE cost function
    """
    m = Y.shape[1]
    return 2 * (A-Y) / m