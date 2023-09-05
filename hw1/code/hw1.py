import numpy as np
from numpy import ndarray


# Question 5.1
def special_reshape(x: ndarray) -> ndarray:
    """
    This function reshapes an input n-dimensional ndarray into a 2-dimensional ndarray
    where the output's first dimension collapses the input's first n-1 dimensions and the
    output's second dimension preserves the last dimension. Here are example input/output
    shapes:
    - Input shape (8, 7) results in output shape (8, 7)
    - Input shape (2, 3, 4) results in output shape (6, 4)
    - Input shape (7, 9, 2, 4, 5) results in output shape (504, 5)
    - Input shape (1, 1, 1, 1, 1, 1) results in output shape (1, 1)

    You may assume that n >= 2.

    Note: A 2-D ndarray of shape (n, 1) is technically different from a 1-D ndarray of
    shape (n,). The output for this question should be the latter.

    Args:
        x (ndarray): Input ndarray.

    Returns:
        ndarray: Output ndarray.
    """
    return x.reshape(-1,x.shape[-1])


# Question 5.2
def linear(x: ndarray, W: ndarray, b) -> ndarray:
    """
    This function performs a linear transformation on an input vector x, given a weight
    matrix W and a bias vector b.

    Args:
        x (ndarray): Input vector of size (m,).
        W (ndarray): Weight matrix of size (n, m).
        b (ndarray): Bias vector of size (n,).

    Returns:
        ndarray: Output vector of size (n,).
    """
    return W@x+b


# Question 5.3
def sigmoid(x: ndarray) -> ndarray:
    """
    This function applies the sigmoid function to an input vector x. If the size of x
    is (n,), this function should return a (n,) ndarray.
    
    Args:
        x (ndarray): Input vector of size (n,).

    Returns:
        ndarray: Output vector of size (n,).
    """
    return 1 / (1 + np.exp(-x))


# Question 5.4
def two_layer_nn(x: ndarray, W1: ndarray, W2: ndarray, b1: ndarray, b2: ndarray) -> ndarray:
    """
    This function simulates a two-layer neural network, given input x, weight matrices
    W1 and W2, and bias vectors b1 and b2. This function should return the output of the
    network.

    Note: You must use linear() and sigmoid() above to implement this function.

    Args:
        x (ndarray): Input vector.
        W1 (ndarray): First layer's weight matrix.
        W2 (ndarray): Second layer's weight matrix.
        b1 (ndarray): First layer's bias vector.
        b2 (ndarray): Second layer's bias vector.

    Returns:
        ndarray: Output vector.
    """
    return sigmoid(linear(sigmoid(linear(x,W1,b1)), W2,b2))
