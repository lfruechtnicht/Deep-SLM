from numpy import zeros

from ..common.activation_functions import ACTIVATION_FUNCTIONS_DICT
from .neuron import Neuron


class OutputNeuron(Neuron):
    """Class representing a neuron of the output layer in a neural network.

    Parameters
    ----------
    activation_function_id : {'identity', 'sigmoid', 'tanh', 'relu'}
        Activation function for the neuron. The available functions are: identity,
        logistic (sigmoid), hyperbolic tangent (tanh), and rectified linear unit (relu).

    semantics : array of shape (num_samples,), optional
        Output vector of neuron after performing all calculations.

    input_connections : array of shape (n_connections,), optional, default None
        Set of connections between this neuron and previous neural_network_components. These connections contain also
        their weights and their state (i.e., if a given connection is activated or deactivated).

    Attributes
    ----------
    semantics : array of shape (num_samples,)
        Output vector of neuron after performing all calculations.

    bias : float
        Value for neuron's bias.

    input_connections : array of shape (n_connections,)
        Connections between this neuron and previous neural_network_components. These connections contain also
        their weights and their state (i.e., if a given connection is activated or deactivated).

    activation_function : function, optional, default None
        Optional activation function for the neuron. The available functions are: identity,
        logistic (sigmoid), hyperbolic tangent (tanh), and rectified linear unit (relu).

    activation_function_id : {'identity', 'sigmoid', 'tanh', 'relu'}
        Id of activation function of the neuron. This information is kept due to copy reasons.
    """

    def __init__(self, activation_function_id, semantics=None, input_connections=None):
        super().__init__(bias=0, semantics=semantics)
        self.bias_increment = 0
        self.activation_function = ACTIVATION_FUNCTIONS_DICT.get(activation_function_id)
        self.activation_function_id = activation_function_id
        self.input_connections = input_connections if input_connections is not None else list()

    def __repr__(self):
        return "OutputNeuron"

    def _calculate_weighted_input(self, num_last_connections=None):
        """Calculates weighted input from input connections."""
        
        instances = self.input_connections[-1].from_neuron.semantics.shape[0]
        weighted_input = zeros(instances)
        
        if num_last_connections:
            weighted_input += self.bias_increment
            connections = self.input_connections[-num_last_connections:]
        else:
            weighted_input += self.bias
            connections = self.input_connections
        
        for connection in filter(lambda x: x.is_active, connections):
            weighted_input += connection.get_from_neuron_semantics() * connection.weight
        
        return weighted_input

    def _calculate_output(self, weighted_input, clip=False):
        """Calculates semantics upon a weighted input.

        Parameters
        ----------
        weighted_input : array of shape (num_samples,)
            Accumulated sum of neural network's semantics from previous layers.
        """
        if clip:
            return self.activation_function(weighted_input).clip(0, 1)
        else:
            return self.activation_function(weighted_input)

    def calculate_semantics(self):
        """Calculates semantics after calculating weighted input."""
        self.weighted_input = self._calculate_weighted_input()
        self.semantics = self._calculate_output(self.weighted_input)
        return self.semantics
    
    def incremental_semantics_update(self, num_last_connections):
        """Adds a given value to current semantics. This function exists to prevent the need
        to recalculate OutputNeuron's semantics everytime a new input connection is added.

        Parameters
        ----------
        partial_semantics : array of shape (num_samples,)
            Partial semantics to be added arising from the addition of new neural_network_components to
            the neural network.
        """
        partial_weighted_input = self._calculate_weighted_input(num_last_connections=num_last_connections)
        
        self.weighted_input += partial_weighted_input
        
        self.semantics = self._calculate_output(self.weighted_input)
        return self.semantics

    def override_semantics(self, semantics):
        """Overrides current neuron's semantics. Usually, this occurs when a clone
        of the neural network is being created.

        Parameters
        ----------
        semantics : float
            New semantics.
        """
        self.semantics = semantics.copy()
    
    def override_weighted_input(self, weighted_input):
        """Overrides current neuron's weighted input. Usually, this occurs when a clone
        of the neural network is being created.

        Parameters
        ----------
        weighted_input : float
            New weighted input.
        """
        self.weighted_input = weighted_input.copy()
        
    def override_input_connections(self, connections):
        self.input_connections = connections

    def add_input_connection(self, new_connection):
        """Receives a new input connection for this neuron.

        Parameters
        ----------
        new_connection : Connection
            New connection with neuron from a previous layer in the neural
            network.
        """
        self.input_connections.append(new_connection)

    def get_activation_function_id(self):
        return self.activation_function_id
    
    def increment_bias(self, increment):
        self.bias_increment = increment
        self.bias += increment
        
    def get_weighted_input(self):
        """Returns current weighted input of the output neuron."""
        return self.weighted_input

    def clean_semantics(self):
        """Resets the semantics and the weighted input."""
        self.semantics = None
        self.weighted_input = None
    
    def free(self):
        self.clean_semantics()
        self.input_connections = None
