from copy import copy

from numpy import zeros, ones, empty, empty_like, where, append as np_append
from scipy.special import xlogy
from sklearn.linear_model import LinearRegression
from sklearn.neural_network._base import softmax

from DeepSemanticLearningMachine.non_convolutional.lbfgs import LBFGS

from .common.neural_network_builder import NeuralNetworkBuilder
from .components.hidden_neuron import HiddenNeuron
from .components.input_neuron import InputNeuron


############################
#### Mutation operators ####
############################
def mutate_hidden_layers(X, y, neural_network, random_state, learning_step, sparseness, maximum_new_neurons_per_layer=3,
                         maximum_neuron_connection_weight=1.0, maximum_bias_weight=1.0,
                         delta_target=None, global_preds=None, hidden_activation_functions_ids=None,
                         prob_activation_hidden_layers=None, time_print=False):
    """
    Function that changes a given neural network's topology by possibly
    adding neurons on each one of the hidden layers.

    Parameters
    ----------
    neural_network : NeuralNetwork
        Neural network to be changed.

    random_state : RandomState instance
        A Numpy random number generator.

    learning_step : float or {'optimized'}
        Strategy to calculate the weight for connections in the last
        hidden layer's neurons to the output layer.

    sparseness : dict
        Dictionary containing information regarding neurons' connections,
        namely sparseness and the existence of skip connections (keys:
        'sparse', 'minimum_sparseness', 'maximum_sparseness', and
        'prob_skip_connection').

    maximum_new_neurons_per_layer : int
        Maximum number of neurons that can be added to a hidden layer.

    maximum_neuron_connection_weight : float
        Maximum value a weight connection between two neurons can have.

    maximum_bias_weight : float
        Maximum value a bias of a neuron can have.

    delta_target : array of shape (num_samples,), optional
        Array containing the distance of neural network's predictions
        to target values. Required if learning_step is set as 'optimized'.

    learning_step_function : function, optional, default None
        Optional function used to calculate optimized learning step
        for the neuron. The available functions are: pinv and lstsq
        from numpy package, and pinv, pinv2 and lstsq from scipy package. ,

    hidden_activation_functions_ids : array of shape (num_functions,), optional
        Names of activation functions that can be used in hidden layers.

    prob_activation_hidden_layers : float, optional
        Probability of the neurons on hidden layers have an activation
        function associated.

    Returns
    -------
    mutated_neural_network : NeuralNetwork
    """

    import timeit
    start_time = timeit.default_timer()
    hidden_semantics_time = 0

    # Add between 0 up to 'maximum_new_neurons_per_layer' neurons in each layer:
    # ===========================================================================
    # new_neurons_per_layer = [random_state.randint(50, 50 + 1) for _ in range(neural_network.get_number_hidden_layers())]
    # ===========================================================================
    new_neurons_per_layer = [random_state.randint(maximum_new_neurons_per_layer, maximum_new_neurons_per_layer + 1) for
                             _ in range(neural_network.get_number_hidden_layers())]
    # ===========================================================================
    # new_neurons_per_layer = [random_state.randint(1, maximum_new_neurons_per_layer) for _ in range(neural_network.get_number_hidden_layers())]
    # ===========================================================================

    # The last hidden layer needs to receive at least 1 new neuron:
    if new_neurons_per_layer[-1] == 0:
        new_neurons_per_layer[-1] = 1

    # Auxiliary list containing neurons to be filtered in skip connections:
    if sparseness.get('prob_skip_connection') > 0:
        neurons_for_skip_connections = copy(neural_network.input_layer)
        neurons_for_skip_connections.extend(neural_network.get_hidden_neurons())
    else:
        neurons_for_skip_connections = None

    # Auxiliary list that will contain references for new hidden neurons created for each layer:
    added_new_neurons = list()

    for i, number_neurons in enumerate(new_neurons_per_layer):

        if number_neurons > 0:
            # Get new hidden layer to extend:
            new_hidden_neurons = NeuralNetworkBuilder.create_hidden_neurons(number_neurons=number_neurons,
                                                                            random_state=random_state, level_layer=i,
                                                                            maximum_bias_weight=maximum_bias_weight,
                                                                            hidden_activation_functions_ids=hidden_activation_functions_ids,
                                                                            prob_activation_hidden_layers=prob_activation_hidden_layers)
            # Establish connections with previous layer:
            if i == 0:
                # Note: Previous layer is the input layer, so there are no skipped connections (although we might have sparseness):
                NeuralNetworkBuilder.connect_layers(layer_to_connect=new_hidden_neurons,
                                                    previous_layer=neural_network.input_layer,
                                                    maximum_neuron_connection_weight=maximum_neuron_connection_weight,
                                                    random_state=random_state,
                                                    sparseness=sparseness, neurons_for_skip_connections=None)
            else:
                # Filter neurons for skip connections:
                if neurons_for_skip_connections:
                    skip_connections_set = list(
                        filter(lambda x: isinstance(x, InputNeuron), neurons_for_skip_connections) if i == 1
                        else list(filter(lambda x: isinstance(x, InputNeuron) or (
                                    isinstance(x, HiddenNeuron) and x.level_layer < i - 1),
                                         neurons_for_skip_connections)))
                else:
                    skip_connections_set = None

                # Note: i-1 is the index of the previous layer:
                NeuralNetworkBuilder.connect_layers(layer_to_connect=new_hidden_neurons,
                                                    previous_layer=neural_network.hidden_layers[i - 1],
                                                    maximum_neuron_connection_weight=maximum_neuron_connection_weight,
                                                    random_state=random_state, sparseness=sparseness,
                                                    neurons_for_skip_connections=skip_connections_set)

                # Starting at second hidden layer (i > 0), all neurons must be connected with at least 1 new neuron added to the previous layer:
                for neuron in new_hidden_neurons:
                    if not any(connection.is_from_previous_layer for connection in neuron.input_connections):
                        # Warning: previous_hidden_layer_index can be None if none of previous layers received new neurons during this mutation.
                        # =======================================================
                        # previous_hidden_layer_index = get_closest_positive_number_index(new_neurons_per_layer, i - 1)
                        # =======================================================
                        previous_hidden_layer_index = 0

                        if previous_hidden_layer_index:
                            NeuralNetworkBuilder.connect_consecutive_mutated_layers(neuron, added_new_neurons[
                                previous_hidden_layer_index],
                                                                                    random_state,
                                                                                    maximum_neuron_connection_weight)

            # Calculate semantics for new hidden neurons:
            hidden_semantics_time_start_time = timeit.default_timer()
            [hidden_neuron.calculate_semantics() for hidden_neuron in new_hidden_neurons]
            hidden_semantics_time += timeit.default_timer() - hidden_semantics_time_start_time

            # Extend new hidden neurons to the respective hidden layer:
            neural_network.extend_hidden_layer(layer_index=i, new_neurons=new_hidden_neurons)

            # Store references of new hidden neurons:
            added_new_neurons.append(new_hidden_neurons)

        else:  # No new hidden neurons were added to this hidden layer:
            added_new_neurons.append(list())

    # Connect new hidden neurons from last hidden layer with output layer:
    NeuralNetworkBuilder.connect_layers(layer_to_connect=neural_network.output_layer,
                                        previous_layer=neural_network.get_last_n_neurons(new_neurons_per_layer[-1]),
                                        maximum_neuron_connection_weight=maximum_neuron_connection_weight,
                                        random_state=random_state,
                                        sparseness={'sparse': False}, neurons_for_skip_connections=None)

    ls_start_time = timeit.default_timer()
    # Calculate learning step for new neurons added in the last hidden layer:

    # ===========================================================================
    # mutation = mutation_ols_ls_margin
    # mutation = mutation_lbfgs_all_neurons
    # ===========================================================================
    mutation = mutation_lbfgs_new_neurons
    mutation(X, y, neural_network, new_neurons_per_layer[-1], random_state)

    # ===========================================================================
    # mutation = None
    # neural_network.calculate_learning_step(learning_step, new_neurons_per_layer[-1], random_state, target, delta_target, learning_step_function)
    # ===========================================================================
    ls_time = timeit.default_timer() - ls_start_time

    incremental_start_time = timeit.default_timer()
    # Sum previous semantics to output layer:
    if mutation == mutation_lbfgs_all_neurons:
        neural_network.calculate_output_semantics()
    else:
        neural_network.incremental_output_semantics_update(num_last_connections=new_neurons_per_layer[-1])
    incremental_time = timeit.default_timer() - incremental_start_time

    time = timeit.default_timer() - start_time
    others_time = time - ls_time - incremental_time - hidden_semantics_time
    if time_print:
        print(
            '\n\t\tmutate_hidden_layers total time = %.3f seconds\n\t\t\tcalculate_learning_step = %.3f seconds\n\t\t\tincremental_output_semantics_update = %.3f seconds\n\t\t\thidden semantics computation = %.3f seconds\n\t\t\tothers = %.3f seconds' % (
            time, ls_time, incremental_time, hidden_semantics_time, others_time))
        print(
            '\n\t\t\tcalculate_learning_step = %% of total mutation time %.2f\n\t\t\tincremental_output_semantics_update = %% of total mutation time %.2f\n\t\t\thidden semantics computation = %% of total mutation time %.2f\n\t\t\tothers = %% of total mutation time %.2f' % (
            ls_time / time * 100, incremental_time / time * 100, hidden_semantics_time / time * 100,
            others_time / time * 100))

    neural_network.new_neurons = added_new_neurons

    # Return mutated neural_network:
    return neural_network


def calculate_ols(nn, num_last_neurons, target):
    instances = target.shape[0]
    partial_semantics = ones((instances, num_last_neurons))

    # Get semantics of last hidden neurons (the same number of last hidden neurons as the number of learning steps to be computed)
    last_hidden_neurons = nn.get_last_n_neurons(num_last_neurons)
    for i, hidden_neuron in enumerate(last_hidden_neurons):
        partial_semantics[:, i] = hidden_neuron.get_semantics()

    for output_index, output_neuron in enumerate(nn.output_layer):

        neuron_targets = target[:, output_index]
        neuron_target_class_1_indices = where(neuron_targets == 1)[0]
        neuron_target_class_1_count = neuron_target_class_1_indices.shape[0]
        neuron_target_class_0_count = instances - neuron_target_class_1_count

        sample_weights = ones(instances)
        class_1_weight = neuron_target_class_0_count / neuron_target_class_1_count
        sample_weights[neuron_target_class_1_indices] = class_1_weight

        reg = LinearRegression().fit(partial_semantics, neuron_targets, sample_weights)
        optimal_weights = np_append(reg.coef_.T, reg.intercept_)

        # Update connections with the learning step value:
        for i in range(num_last_neurons):
            output_neuron.input_connections[-num_last_neurons + i].weight = optimal_weights[i]

        output_neuron.increment_bias(optimal_weights[-1])


def mutation_ols_ls_margin(X, y, nn, n_new_neurons, random_state):
    n_samples = y.shape[0]
    hidden_semantics = zeros((n_samples, n_new_neurons))
    for i, hidden_neuron in enumerate(nn.get_last_n_neurons(n_new_neurons)):
        hidden_semantics[:, i] = hidden_neuron.get_semantics()

    y_prob = nn.get_predictions().copy()
    softmax(y_prob)
    ce_loss = -xlogy(y, y_prob)
    m = ce_loss.max(axis=1)
    # am = ce_loss.argmax(axis=1)

    sample_weights = 1 + m
    # sample_weights = ones(n_samples)

    for output_index, output_neuron in enumerate(nn.output_layer):
        output_delta_y = y[:, output_index] - nn.get_predictions()[:, output_index]
        reg = LinearRegression().fit(hidden_semantics, output_delta_y, sample_weights)
        optimal_weights = np_append(reg.coef_.T, reg.intercept_)
        # print('\n\toptimal_weights [min, mean, max]: [%.5f, %.5f, %.5f]' % (optimal_weights.min(), optimal_weights.mean(), optimal_weights.max()))

        # Update connections with the learning step value:
        for i in range(n_new_neurons):
            output_neuron.input_connections[-n_new_neurons + i].weight = optimal_weights[i]

        output_neuron.increment_bias(optimal_weights[-1])


def mutation_lbfgs_new_neurons(X, y, nn, n_new_neurons, random_state):
    n_samples = y.shape[0]
    new_neurons = nn.get_last_n_neurons(n_new_neurons)
    hidden_semantics = zeros((n_samples, n_new_neurons))
    for i, hidden_neuron in enumerate(new_neurons):
        hidden_semantics[:, i] = hidden_neuron.get_semantics()

    layer_units = [n_new_neurons, y.shape[1]]
    activations = []
    activations.extend([hidden_semantics])
    activations.extend(empty((n_samples, n_fan_out)) for n_fan_out in layer_units[1:])
    deltas = [empty_like(a_layer) for a_layer in activations]
    coef_grads = [empty((n_fan_in_, n_fan_out_)) for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])]
    intercept_grads = [empty(n_fan_out_) for n_fan_out_ in layer_units[1:]]

    solver = LBFGS()

    """ zero-weight initialization for new neurons """
    coef_init = zeros((layer_units[0], layer_units[1]))

    intercept_init = zeros(layer_units[1])
    for output_index, output_neuron in enumerate(nn.output_layer):
        intercept_init[output_index] = output_neuron.bias

    fixed_weighted_input = zeros((n_samples, layer_units[1]))
    for output_index, output_neuron in enumerate(nn.output_layer):
        fixed_weighted_input[:, output_index] = output_neuron.get_weighted_input() - output_neuron.bias

    coefs, intercepts = solver.fit(X, y, activations, deltas, coef_grads, intercept_grads, layer_units, random_state,
                                   coef_init=coef_init, intercept_init=intercept_init,
                                   fixed_weighted_input=fixed_weighted_input)

    coefs = coefs[-1]
    intercepts = intercepts[-1]
    for output_index, output_neuron in enumerate(nn.output_layer):
        for i in range(n_new_neurons):
            # print('coefs[%d, %d] = %.5f\n' % (i, output_index, coefs[i, output_index]))
            output_neuron.input_connections[-n_new_neurons + i].weight = coefs[i, output_index]

        # print('intercepts[%d] = %.5f\n' % (output_index, intercepts[output_index]))
        # output_neuron.bias = intercepts[output_index]
        output_neuron.increment_bias(intercepts[output_index] - output_neuron.bias)


def mutation_lbfgs_all_neurons(X, y, nn, n_new_neurons, random_state):
    n_samples = y.shape[0]
    n_neurons = len(nn.hidden_layers[-1])
    hidden_semantics = zeros((n_samples, n_neurons))
    for i, hidden_neuron in enumerate(nn.hidden_layers[-1]):
        hidden_semantics[:, i] = hidden_neuron.get_semantics()

    layer_units = [n_neurons, y.shape[1]]
    activations = []
    activations.extend([hidden_semantics])
    activations.extend(empty((n_samples, n_fan_out)) for n_fan_out in layer_units[1:])
    deltas = [empty_like(a_layer) for a_layer in activations]
    coef_grads = [empty((n_fan_in_, n_fan_out_)) for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])]
    intercept_grads = [empty(n_fan_out_) for n_fan_out_ in layer_units[1:]]

    """ zero-weight initialization for new neurons """
    coef_init = zeros((layer_units[0], layer_units[1]))
    for output_index, output_neuron in enumerate(nn.output_layer):
        for i, connection in enumerate(output_neuron.input_connections[:-n_new_neurons]):
            coef_init[i][output_index] = connection.weight
            # print("i =", i, ", output_index =", output_index, ", weight =", connection.weight)

    intercept_init = zeros(layer_units[1])
    for output_index, output_neuron in enumerate(nn.output_layer):
        intercept_init[output_index] = output_neuron.bias

    solver = LBFGS()
    coefs, intercepts = solver.fit(X, y, activations, deltas, coef_grads, intercept_grads, layer_units, random_state,
                                   coef_init=coef_init, intercept_init=intercept_init)
    coefs = coefs[-1]
    intercepts = intercepts[-1]
    for output_index, output_neuron in enumerate(nn.output_layer):
        for i in range(n_neurons):
            # print('coefs[%d, %d] = %.5f\n' % (i, output_index, coefs[i, output_index]))
            output_neuron.input_connections[-n_neurons + i].weight = coefs[i, output_index]

        # print('intercepts[%d] = %.5f\n' % (output_index, intercepts[output_index]))
        output_neuron.bias = intercepts[output_index]
