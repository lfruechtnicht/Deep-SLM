from numpy import exp, tanh
from sklearn.neural_network._base import relu


##################################################
#### Activation functions for neural networks ####
##################################################
def apply_identity(input_sum_array):
    """Simply returns the input array.

    Parameters
    ----------
    input_sum_array : {array-like, sparse matrix}, shape (n_samples, n_features)
        Data, where n_samples is the number of samples
        and n_features is the number of features.

    Returns
    -------
    input_sum_array : {array-like, sparse matrix}, shape (n_samples, n_features)
        Same as the input data.
    """
    return input_sum_array


def apply_sigmoid(input_sum_array):
    """Computes the logistic sigmoid function inplace.

    Parameters
    ----------
    input_sum_array : {array-like, sparse matrix}, shape (n_samples, n_features)
        Data, where n_samples is the number of samples
        and n_features is the number of features.

    Returns
    -------
    input_sum_array : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    return 1 / (1 + exp(-input_sum_array))


def apply_tanh(input_sum_array):
    """Computes the hyperbolic tanh function inplace.

    Parameters
    ----------
    input_sum_array : {array-like, sparse matrix}, shape (n_samples, n_features)
        Data, where n_samples is the number of samples
        and n_features is the number of features.

    Returns
    -------
    input_sum_array : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    return tanh(input_sum_array)


def apply_relu(input_sum_array):
    """Computes the rectified linear unit function inplace.

    Parameters
    ----------
    input_sum_array : {array-like, sparse matrix}, shape (n_samples, n_features)
        Data, where n_samples is the number of samples
        and n_features is the number of features.

    Returns
    -------
    input_sum_array : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    return relu(input_sum_array)


#######################################################################################
#### Dictionaries of activation functions and method to select activation function ####
#######################################################################################

ACTIVATION_FUNCTIONS_DICT = {
    'identity': apply_identity,
    'sigmoid': apply_sigmoid,
    'tanh': apply_tanh,
    'relu': apply_relu
}

NON_LINEAR_ACTIVATION_FUNCTIONS_DICT = {
    'sigmoid': apply_sigmoid,
    'tanh': apply_tanh,
    'relu': apply_relu
}
