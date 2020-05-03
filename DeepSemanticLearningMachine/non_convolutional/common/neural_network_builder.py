"""
This file contains a class called NeuralNetworkBuilder that holds static methods to build neural networks or some of their components.

Warning: None of these methods is responsible for mathematical calculations inside the neural network (such as calculating semantics, for example).
"""
from numpy import delete as np_delete

from ..components.connection import Connection
from ..components.hidden_neuron import HiddenNeuron
from ..components.input_neuron import InputNeuron
from ..components.output_neuron import OutputNeuron
from .neural_network import NeuralNetwork


##############################################################
#### Functions used to create Neurons and Neural Networks ####
##############################################################
class NeuralNetworkBuilder:
    """Class that encapsulates static methods used to generate fully
    functional neural networks, as well as loose components, such as
    hidden layers, and so on.
    """

    @staticmethod
    def generate_new_neural_network(number_hidden_layers,
                                    number_hidden_neurons,
                                    number_output_neurons,
                                    maximum_neuron_connection_weight,
                                    maximum_bias_weight,
                                    output_activation_function_id,
                                    input_layer,
                                    random_state,
                                    sparseness,
                                    hidden_activation_functions_ids=None,
                                    prob_activation_hidden_layers=None):
        """
        Generates a complete and functional neural network.

        Parameters
        ----------
        number_hidden_layers : int
            Number of hidden layers neural network should have.

        number_hidden_neurons : array of shape (number_hidden_layers,)
            Number of neurons that each hidden layer should have.

        number_output_neurons: int
            Number of neurons that output layer of the neural network
            should have.

        maximum_neuron_connection_weight : float
            Maximum value a weight connection between two neurons can have.

        maximum_bias_weight : float
            Maximum value a bias of a neuron can have.

        output_activation_function_id : string
            Name of activation function for neurons in output layer.

        input_layer : ...
            ...

        random_state : RandomState instance
            A Numpy random number generator.

        sparseness : dict
            Dictionary containing information regarding neurons' connections, namely
            sparseness and the existence of skip connections (keys: 'sparse',
            'minimum_sparseness', 'maximum_sparseness', and 'prob_skip_connection').

        hidden_activation_functions_ids : array of shape (num_functions,), optional
            Names of activation functions that can be used in hidden layers.

        prob_activation_hidden_layers : float, optional
            Probability of the neurons on hidden layers have an activation
            function associated.

        Returns
        -------
        neural_network : NeuralNetwork
            Newly created neural network.
        """

        # Create a list with N lists of hidden neurons:
        hidden_layers = list()
        for i in range(number_hidden_layers):
            number_neurons = number_hidden_neurons[i]
            hidden_layer = NeuralNetworkBuilder.create_hidden_neurons(number_neurons, random_state, i,
                                                                      maximum_bias_weight,
                                                                      hidden_activation_functions_ids,
                                                                      prob_activation_hidden_layers)
            hidden_layers.append(hidden_layer)

        # Create list with output neurons:
        output_layer = NeuralNetworkBuilder.create_output_neurons(number_neurons=number_output_neurons,
                                                                  activation_function_id=output_activation_function_id)

        # Build connections of neural network:
        input_layer, hidden_layers, output_layer = NeuralNetworkBuilder.connect_neural_network(input_layer,
                                                                                               hidden_layers,
                                                                                               output_layer,
                                                                                               maximum_neuron_connection_weight,
                                                                                               random_state,
                                                                                               sparseness)

        # Instantiate Neural Network:
        neural_network = NeuralNetwork(input_layer, hidden_layers, output_layer)

        # Compute semantics for hidden neurons:
        for hidden_layer in neural_network.hidden_layers:
            # print('\t\tCall to hidden_neuron.calculate_semantics()')
            [hidden_neuron.calculate_semantics() for hidden_neuron in hidden_layer]

        return neural_network

    @staticmethod
    def create_input_neurons(X):
        """Generates the required number of neurons for a neural network's
        input layer considering the dimensions of a given input data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input data. This parameter is required to load the data.

        Returns
        -------
        input_layer : array of shape (num_input_neurons,)
            Layer containing all input neurons.
        """
        input_neurons = list()
        for input_data in X.T:
            input_neurons.append(InputNeuron(input_data))
        return input_neurons

    @staticmethod
    def create_hidden_neurons(number_neurons, random_state, level_layer, maximum_bias_weight=1.0,
                              hidden_activation_functions_ids=None,
                              prob_activation_hidden_layers=None):
        """Generates a given number of neurons for a neural network's hidden layer.

        Parameters
        ----------
        number_neurons : int
            Number of neurons for hidden layer.

        random_state : RandomState instance
            A Numpy random number generator..

        level_layer : int
            Index of hidden layer where this neuron is being inserted.

        maximum_bias_weight : float
            Maximum value a bias of a neuron can have.

        hidden_activation_functions_ids : array of shape (num_functions,), optional
            Names of activation functions that can be used in hidden layers.

        prob_activation_hidden_layers : float, optional
            Probability of the neurons on hidden layers have an activation
            function associated.

        Returns
        -------
        hidden_layer : array of shape (num_hidden_neurons,)
            Hidden layer containing the required number of neurons.
        """
        hidden_neurons = list()
        for _ in range(number_neurons):
            bias = random_state.uniform(-maximum_bias_weight, maximum_bias_weight)
            # Select randomly an activation function with a given probability:
            if prob_activation_hidden_layers is not None and random_state.uniform() <= prob_activation_hidden_layers:
                activation_function_id = hidden_activation_functions_ids[
                    random_state.randint(0, len(hidden_activation_functions_ids))]
            else:
                activation_function_id = None

            hidden_neurons.append(HiddenNeuron(bias, level_layer, activation_function_id=activation_function_id))

        return hidden_neurons

    @staticmethod
    def create_output_neurons(number_neurons, activation_function_id):
        """Generates a given number of neurons for a neural network's output layer.

        Parameters
        ----------
        number_neurons : int
            Number of neurons for output layer.

        activation_function_id : string
            Name of activation function for neurons in output layer.

        Returns
        -------
        output_layer : array of shape (num_output_neurons,)
            Output layer containing the required number of neurons.
        """
        output_neurons = list()
        for _ in range(number_neurons):
            output_neurons.append(OutputNeuron(activation_function_id=activation_function_id))

        return output_neurons

    @staticmethod
    def connect_neural_network(input_layer, hidden_layers, output_layer, maximum_neuron_connection_weight, random_state,
                               sparseness):
        """Establishes all connections of a neural network's layers.

        Parameters
        ----------
        input_layer : array of shape (num_input_neurons,)
            Input layer of a neural network.

        hidden_layers : array of shape (num_hidden_layers,)
            Set of hidden layers of a neural network.

        output_layer : array of shape (num_output_neurons,)
            Output layer of a neural network.

        maximum_neuron_connection_weight : float
            Maximum value a weight connection between two neurons can have.

        random_state : RandomState instance
            A Numpy random number generator.

        sparseness : dict
            Dictionary containing information regarding neurons' connections, namely
            sparseness and the existence of skip connections (keys: 'sparse',
            'minimum_sparseness', 'maximum_sparseness', and 'prob_skip_connection').

        Returns
        -------
        input_layer : array of shape (num_input_neurons,)
            Input layer with connections for the first hidden layer.

        hidden_layers : array of shape (num_hidden_layers,)
            Set of hidden layers with connections among each other. The last
            hidden layer is fully connected to the output layer.

        output_layer : array of shape (num_output_neurons,)
            Output layer with connections from the last hidden layer.
        """

        # Connect input layer with first hidden layer (which is hidden_layers[0]):
        hidden_layers[0] = NeuralNetworkBuilder.connect_layers(layer_to_connect=hidden_layers[0],
                                                               previous_layer=input_layer,
                                                               maximum_neuron_connection_weight=maximum_neuron_connection_weight,
                                                               random_state=random_state, sparseness=sparseness,
                                                               neurons_for_skip_connections=None)

        # Auxiliary list that will contain neurons used for skip connections:
        neurons_for_skip_connections = [] if sparseness.get('prob_skip_connection') > 0 else None

        # Connect second hidden layer and remaining ones:
        for i, hidden_layer in enumerate(hidden_layers[1:]):
            # Collect new neurons for skip connections:
            if sparseness.get('prob_skip_connection') > 0:
                neurons_for_skip_connections.extend(input_layer) if i == 0 else neurons_for_skip_connections.extend(
                    hidden_layers[i - 1])

            # Note: since i starts on zero, hidden_layers[i] is always the reference to the previous layer.
            hidden_layer = NeuralNetworkBuilder.connect_layers(layer_to_connect=hidden_layer,
                                                               previous_layer=hidden_layers[i],
                                                               maximum_neuron_connection_weight=maximum_neuron_connection_weight,
                                                               random_state=random_state,
                                                               sparseness=sparseness,
                                                               neurons_for_skip_connections=neurons_for_skip_connections)

        # Connect last hidden layer with output layer (there is no sparseness, hence 'sparse' is False), and there are no skipped connections:
        output_layer = NeuralNetworkBuilder.connect_layers(layer_to_connect=output_layer,
                                                           previous_layer=hidden_layers[-1],
                                                           maximum_neuron_connection_weight=maximum_neuron_connection_weight,
                                                           random_state=random_state,
                                                           sparseness={'sparse': False},
                                                           neurons_for_skip_connections=None)

        # Return layers after being connected:
        return input_layer, hidden_layers, output_layer

    @staticmethod
    def connect_layers(layer_to_connect, previous_layer, maximum_neuron_connection_weight, random_state, sparseness,
                       neurons_for_skip_connections=None):
        """Establishes the connections amongst two layers in a neural network.

        Parameters
        ----------
        layer_to_connect : array of shape (num_neurons,)
            Layer to connect with another.

        previous_layer : array of shape (num_neurons,)
            Layer to be connected with.

        maximum_neuron_connection_weight : float
            Maximum value a weight connection between two neurons can have.

        random_state : RandomState instance
            A Numpy random number generator.

        sparseness : dict
            Dictionary containing information regarding neurons' connections, namely
            sparseness and the existence of skip connections (keys: 'sparse',
            'minimum_sparseness', 'maximum_sparseness', and 'prob_skip_connection'
            or 'sparse').
            Note: if sparseness dictionary contains a single key ('sparse') means
            that layer_to_connect is neural network's output layer.

        neurons_for_skip_connections : array of shape (num_neurons,), optional, default None
            Set of neurons from previous layers (except from neural network's
            immediately preceding layer) that can be connected or not
            with 'neuron'. When None, it means that we are either connecting the
            first hidden layer to the input layer, or skipping connections are not
            being performed.

        Returns
        -------
        layer_to_connect : array of shape (num_neurons,)
            Layer with connections established with a previous layer.
        """
        if sparseness.get('sparse') is True:
            [NeuralNetworkBuilder._make_sparse_connections(neuron, previous_layer, random_state,
                                                           maximum_neuron_connection_weight, sparseness,
                                                           neurons_for_skip_connections) for neuron in layer_to_connect]
        else:  # Fully-connected layers:
            [NeuralNetworkBuilder._make_fully_connections(neuron, previous_layer, random_state,
                                                          maximum_neuron_connection_weight,
                                                          sparseness.get('prob_skip_connection'),
                                                          neurons_for_skip_connections) for neuron in layer_to_connect]

        # Return layer already connected:
        return layer_to_connect

    @staticmethod
    def connect_consecutive_mutated_layers(neuron, previous_layer, random_state, maximum_neuron_connection_weight):
        """Establishes a connection amongst two consecutive mutated layers in a
        neural network, in order to ensure that information generated by new
        neurons added to a neural network doesn't get lost. Before setting the
        new connection, it randomly eliminates one existing connection to maintain
        sparseness properties of the neural network.

        Parameters
        ----------
        neuron : Neuron
            Neuron to connect with a previous layer.

        previous_layer : array of shape (num_neurons,)
            Set of neurons that can be connected or not with 'neuron'.

        maximum_neuron_connection_weight : float
            Maximum value a weight connection between two neurons can have.

        random_state : RandomState instance
            A Numpy random number generator.
        """
        # Delete (randomly) existent connection from 'neuron':
        neuron.input_connections.remove(random_state.choice(neuron.input_connections, 1))

        previous_neuron = previous_layer[random_state.randint(0, len(previous_layer))]

        NeuralNetworkBuilder._connect_neurons(previous_neuron, neuron,
                                              random_state.uniform(-maximum_neuron_connection_weight,
                                                                   maximum_neuron_connection_weight), True, True)

    @staticmethod
    def _make_sparse_connections(neuron, previous_layer, random_state, maximum_neuron_connection_weight, sparseness,
                                 neurons_for_skip_connections=None):
        """Sparse connections mean that a given neuron is able to not set some
        connections for a given layer.

        Parameters
        ----------
        neuron : Neuron
            Neuron to connect with a previous layer.

        previous_layer : array of shape (num_neurons,)
            Set of neurons that can be connected or not with 'neuron'.

        maximum_neuron_connection_weight : float
            Maximum value a weight connection between two neurons can have.

        random_state : RandomState instance
            A Numpy random number generator.

        sparseness : dict
            Dictionary containing information regarding neurons' connections, namely
            sparseness and the existence of skip connections (keys: 'sparse',
            'minimum_sparseness', 'maximum_sparseness', and 'prob_skip_connection').

        neurons_for_skip_connections : array of shape (num_neurons,), optional, default None
            Set of neurons from previous layers (except from neural network's
            immediately preceding layer) that can be connected or not
            with 'neuron'. When None, it means that we are either connecting the
            first hidden layer to the input layer, or skipping connections are not
            being performed.
        """
        # Get neuron's sparseness, i.e., from the total number of possible connections define randomly the number of connections to activate:
        proportion_sparse_connections = random_state.uniform(sparseness.get('minimum_sparseness'),
                                                             sparseness.get('maximum_sparseness'))
        # Note: it's necessary to guarantee at least 1 connection
        num_connections = max(round(len(previous_layer) * (1 - proportion_sparse_connections)), 1)
        # Get neurons of previous layer to connect with current neuron (sample without replacement):
        neurons_to_connect = random_state.choice(previous_layer, num_connections, replace=False)

        # =======================================================================
        # if neurons_for_skip_connections:
        #     print('len(neurons_for_skip_connections) vs. len(neurons_to_connect) =', len(neurons_for_skip_connections), ' vs.', len(neurons_to_connect))
        # =======================================================================

        # Connect neurons:
        if neurons_for_skip_connections:
            for _ in range(num_connections):
                prob_skip_connection = random_state.uniform()

                # ===============================================================
                # if len(neurons_for_skip_connections) == 0:
                #     print('\t\t\t[Debug] len(neurons_for_skip_connections) == 0')
                # ===============================================================
                if len(neurons_to_connect) == 0:
                    print('\t\t\t[Debug] len(neurons_to_connect) == 0')

                if len(neurons_for_skip_connections) > 0 and prob_skip_connection <= sparseness.get(
                        'prob_skip_connection'):
                    # Get random index for previous_neuron:
                    previous_neuron_id = NeuralNetworkBuilder._raffle_neuron_id(random_state, 0,
                                                                                len(neurons_for_skip_connections))
                    # Collect neuron and remove it from the list to avoid future duplicate connections:
                    previous_neuron = neurons_for_skip_connections[previous_neuron_id]
                    neurons_for_skip_connections = np_delete(neurons_for_skip_connections, previous_neuron_id)
                    # Set flag regarding previous_layer:
                    is_from_previous_layer = False
                else:
                    # Get random index for previous_neuron:
                    previous_neuron_id = NeuralNetworkBuilder._raffle_neuron_id(random_state, 0,
                                                                                len(neurons_to_connect))
                    # Collect neuron and remove it from the list to avoid future duplicate connections:
                    previous_neuron = neurons_to_connect[previous_neuron_id]
                    neurons_to_connect = np_delete(neurons_to_connect, previous_neuron_id)
                    # Set flag regarding previous_layer:
                    is_from_previous_layer = True

                # Connect the two neurons:
                NeuralNetworkBuilder._connect_neurons(previous_neuron, neuron,
                                                      random_state.uniform(-maximum_neuron_connection_weight,
                                                                           maximum_neuron_connection_weight), True,
                                                      is_from_previous_layer)
        else:  # There are no connections to be skipped:
            [NeuralNetworkBuilder._connect_neurons(previous_neuron, neuron,
                                                   random_state.uniform(-maximum_neuron_connection_weight,
                                                                        maximum_neuron_connection_weight), True, True)
             for previous_neuron in neurons_to_connect]

    @staticmethod
    def _make_fully_connections(neuron, previous_layer, random_state, maximum_neuron_connection_weight,
                                prob_skip_connection=0.0, neurons_for_skip_connections=None):
        """Connects a neuron with all other neurons from a given neural network's layer.

        Parameters
        ----------
        neuron : Neuron
            Neuron to connect with a previous layer.

        previous_layer : array of shape (num_neurons,)
            Set of neurons that can be connected or not with 'neuron'.

        random_state : RandomState instance
                    A Numpy random number generator.

        maximum_neuron_connection_weight : float
            Maximum value a weight connection between two neurons can have.

        prob_skip_connection : float, optional, default 0.0
            Probability of the neurons on hidden layers establish a connection with another
            neuron of a layer other than the previous one.

        neurons_for_skip_connections : array of shape (num_neurons,), optional, default None
            Set of neurons from previous layers (except from neural network's
            immediately preceding layer) that can be connected or not
            with 'neuron'. When None, it means that we are either connecting the
            first hidden layer to the input layer, or skipping connections are not
            being performed.
        """
        # Verify if skip connections can occur:
        if neurons_for_skip_connections:
            # Make safety copy of previous layer:
            for i in range(len(previous_layer)):
                generated_prob_skip_connection = random_state.uniform()

                if generated_prob_skip_connection <= prob_skip_connection:
                    # Get random index for previous_neuron:
                    previous_neuron_id = NeuralNetworkBuilder._raffle_neuron_id(random_state, 0,
                                                                                len(neurons_for_skip_connections))
                    previous_neuron = neurons_for_skip_connections[previous_neuron_id]
                    # Remove previous neuron from neurons_for_skip_connections to avoid future duplicated connections:
                    neurons_for_skip_connections = np_delete(neurons_for_skip_connections, previous_neuron_id)
                    # Set flag regarding previous_layer:
                    is_from_previous_layer = False
                else:
                    previous_neuron = previous_layer[i]
                    # Set flag regarding previous_layer:
                    is_from_previous_layer = True
                NeuralNetworkBuilder._connect_neurons(previous_neuron, neuron,
                                                      random_state.uniform(-maximum_neuron_connection_weight,
                                                                           maximum_neuron_connection_weight), True,
                                                      is_from_previous_layer)

        # There are no skip connections to be taken into account:
        else:

            [NeuralNetworkBuilder._connect_neurons(previous_neuron, neuron,
                                                   random_state.uniform(-maximum_neuron_connection_weight,
                                                                        maximum_neuron_connection_weight), True, True)
             for previous_neuron in previous_layer]

    @staticmethod
    def _connect_neurons(first_neuron, second_neuron, weight, is_active_connection, is_from_previous_layer):
        """Establishes the connection between two neurons. The first_neuron is inserted
        into second_neuron's input_connections set.

        Parameters
        ----------
        first_neuron : Neuron
            Neuron to connect.

        second_neuron : Neuron
            Neuron to be connected with.

        weight : float
            Weight / value of the connection.

        is_active_connection : bool
            Determines if this connection should be taken into consideration when
            computing a neural network's semantics.

        is_from_previous_layer : bool
            Determines if this connection is set between two neurons of consecutive
            layers.
        """
        connection = Connection(from_neuron=first_neuron, to_neuron=second_neuron, weight=weight,
                                is_active=is_active_connection)
        second_neuron.add_input_connection(connection)

    @staticmethod
    def clone_neural_network(neural_network, time_print=False):
        """Creates a copy of a given neural network by
        keeping some references to original neural network.

        Parameters
        ----------
        neural_network : NeuralNetwork
            Neural network to be cloned.

        Returns
        -------
            clone : NeuralNetwork
                Cloned neural network.
        """

        import timeit
        start_time = timeit.default_timer()

        # Recreate hidden layers and copy original values:
        hidden_layers = [[neuron for neuron in hidden_layer] for hidden_layer in neural_network.hidden_layers]
        hidden_neurons_list_time = timeit.default_timer() - start_time

        # Recreate output layer and copy original values:
        output_layer = list()
        output_neurons = NeuralNetworkBuilder.create_output_neurons(number_neurons=len(neural_network.output_layer),
                                                                    activation_function_id=neural_network.output_layer[
                                                                        0].get_activation_function_id())

        connections_creation_sum = 0
        for cloned_output_neuron, output_neuron in zip(output_neurons, neural_network.output_layer):
            cloned_output_neuron.override_semantics(output_neuron.get_semantics())
            cloned_output_neuron.override_weighted_input(output_neuron.get_weighted_input())

            cloned_output_neuron.bias = output_neuron.bias
            cloned_output_neuron.bias_increment = output_neuron.bias_increment

            connections_creation_start = timeit.default_timer()
            # A
            connections = [connection for connection in output_neuron.input_connections]
            cloned_output_neuron.override_input_connections(connections)
            # B
            # ===================================================================
            # for connection in output_neuron.input_connections:
            #     from_neuron = connection.from_neuron
            #     to_neuron = cloned_output_neuron
            #     weight = connection.weight
            #     Connection(from_neuron, to_neuron, weight)
            # ===================================================================
            connections_creation_sum += timeit.default_timer() - connections_creation_start

            output_layer.append(cloned_output_neuron)

        # Note: The input layer does not need to be recreated because its values will not change in an immediate future.
        clone = NeuralNetwork(neural_network.input_layer, hidden_layers, output_layer)
        clone.update_parent(neural_network.is_better_than_parent())
        clone.update_loss(neural_network.get_loss())
        clone.override_predictions(neural_network.get_predictions())

        time = timeit.default_timer() - start_time
        if time_print:
            print('\n\t\tclone_neural_network = %.3f seconds' % (time))
            print('\n\t\t\thidden neurons list creation = %.3f seconds' % (hidden_neurons_list_time))
            print('\n\t\t\tconnections creation = %.3f seconds' % (connections_creation_sum))

        return clone

    @staticmethod
    def _raffle_neuron_id(random_state, minimum_value, maximum_value):
        """Generates a random integer number to be used as an
        index to select a neuron from a given layer."""
        return random_state.randint(minimum_value, maximum_value)
