from numpy import zeros

from ..common.activation_functions import ACTIVATION_FUNCTIONS_DICT
from .neuron import Neuron


class HiddenNeuron(Neuron):
    """Class representing a neuron of a hidden layer in a neural network.

    Parameters
    ----------
    bias : float
        Value for neuron's bias.

    level_layer : int
        Index of hidden layer where this neuron is being inserted.

    semantics : array of shape (num_samples,), optional
        Output vector of neuron after performing all calculations.

    input_connections : array of shape (n_connections,), optional, default None
        Set of connections between this neuron and previous neurons. These connections contain also
        their weights and their state (i.e., if a given connection is activated or deactivated).

    activation_function_id : {'identity', 'sigmoid', 'tanh', 'relu'}, optional, default None
        Optional activation function for the neuron. The available functions are: identity,
        logistic (sigmoid), hyperbolic tangent (tanh), and rectified linear unit (relu).

    Attributes
    ----------
    semantics : array of shape (num_samples,)
        Output vector of neuron after performing all calculations.

    bias : float
        Value for neuron's bias.

    input_connections : array of shape (n_connections,)
        Connections between this neuron and previous neurons. These connections contain also
        their weights and their state (i.e., if a given connection is activated or deactivated).

    activation_function : function, optional, default None
        Optional activation function for the neuron. The available functions are: identity,
        logistic (sigmoid), hyperbolic tangent (tanh), and rectified linear unit (relu).

    level_layer : int
        Index of hidden layer where this neuron is inserted.
    """

    def __init__(self, bias, level_layer, semantics=None, input_connections=None, activation_function_id='identity'):
        super().__init__(bias=bias, semantics=semantics)
        self.input_connections = input_connections if input_connections is not None else list()
        self.activation_function = ACTIVATION_FUNCTIONS_DICT.get(activation_function_id)
        self.level_layer = level_layer

    def __repr__(self):
        return "HiddenNeuron"

    def _calculate_weighted_input(self):
        """Calculates weighted input from input connections."""
        
        instances = self.input_connections[0].from_neuron.semantics.shape[0]
        weighted_input = zeros(instances)
        
        weighted_input += self.bias
        
        for connection in filter(lambda x: x.is_active, self.input_connections):
            weighted_input += connection.get_from_neuron_semantics() * connection.weight
        
        return weighted_input

    def _calculate_output(self, weighted_input):
        """Calculates semantics upon a weighted input.

        Parameters
        ----------
        weighted_input : array of shape (num_samples,)
            Accumulated sum of neural network's semantics from previous layers.
        """
        if self.activation_function:
            return self.activation_function(weighted_input)
        else:
            return weighted_input

    def calculate_semantics(self):
        """Calculates semantics after calculating weighted input."""
        #=======================================================================
        # print('\t\t\tCalculating semantics for hidden neuron with %d input connections' % (len(self.input_connections)))
        #=======================================================================
        weighted_input = self._calculate_weighted_input()
        self.semantics = self._calculate_output(weighted_input)
        weighted_input = None
    
    def free(self):
        self.clean_semantics()
        self.input_connections = None
    
    def add_input_connection(self, new_connection):
        """Receives a new input connection for this neuron.

        Parameters
        ----------
        new_connection : Connection
            New connection with neuron from a previous layer in the neural
            network.
        """
        self.input_connections.append(new_connection)
