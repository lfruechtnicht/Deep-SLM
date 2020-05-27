from .hidden_neuron import HiddenNeuron
from .input_neuron import InputNeuron
from .output_neuron import OutputNeuron


class Connection:
    """Class representing the connection between two neurons in a neural network.

    Parameters
    ----------
    from_neuron : InputNeuron or HiddenNeuron
        Neuron to be set as origin of the connection.

    to_neuron : HiddenNeuron or OutputNeuron
        Neuron to be set as destination of the connection by adding from_neuron to
        its collection of input connections.

    weight : float
        Value of the connection.

    is_active : bool
        Flag that indicates if this connection is being considered in the neural network.
        If True, it indicates that this connection should be considered when computing
        semantics for the to_neuron.

    Attributes
    ----------
        from_neuron: InputNeuron or HiddenNeuron
            Origin of the connection.

        weight: float
            Weight's value of the connection.

        is_active: bool
            Flag that indicates if this connection is being considered in the neural network.

        is_from_previous_layer : bool
            Flag that indicates if this connection is set between two consecutive layers.
    """

    def __init__(self, from_neuron, to_neuron, weight, is_active=True):
        # Verify if connection can be made:
        if self._validate_connection(from_neuron, to_neuron):
            # Proceed with connection:
            self.from_neuron = from_neuron
            self.to_neuron = to_neuron
            self.weight = weight
            self.is_active = is_active

            self.is_from_previous_layer = self._is_from_previous_neuron()

        else:
            # Throw error saying that is not possible to connect these neural_network_components (TODO for 2nd commit)
            # Note: This else statement will be removed in the final version of the package. For now, it will be used for debugging purposes.
            print('\t\t\t[Debug] Invalid connection')

    def __repr__(self):
        return "Connection"

    def get_from_neuron_semantics(self):
        """ """
        return self.from_neuron.get_semantics()

    def _validate_connection(self, from_neuron, to_neuron):
        """ Makes sure the connection between two neural_network_components can be made, avoiding situations such as an output
        neuron connecting backwards with another neuron, or an Input Neuron receiving a connection from another
        neuron, or if the connection has already been made.

        Parameters
        ----------
        from_neuron : Neuron, derived class
            Origin of the connection between two neural_network_components being validated.

        to_neuron : Neuron, derived class
            Destination of the connection between two neural_network_components being validated.

        Return
        ------
        connection_viability : bool
        """
        return True

    def _is_from_previous_neuron(self):
        """ """
        if isinstance(self.from_neuron, InputNeuron) and self.to_neuron.level_layer == 0:
            return True

        if isinstance(self.from_neuron, HiddenNeuron) \
                and isinstance(self.to_neuron, HiddenNeuron) \
                and self.from_neuron.level_layer == (self.to_neuron.level_layer - 1):
            return True

        if isinstance(self.to_neuron, OutputNeuron):
            return True  # Note: These type of neurons only establish connections with neurons from the last hidden layer.

        return False
