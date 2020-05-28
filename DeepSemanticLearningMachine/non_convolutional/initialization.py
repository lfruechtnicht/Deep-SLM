from numpy import zeros, ones, empty, empty_like, where, append as np_append
from sklearn.linear_model import LinearRegression

from DeepSemanticLearningMachine.non_convolutional.lbfgs import LBFGS


def init_ols_balanced(X, y, nn, random_state):
    n_samples = y.shape[0]
    n_neurons = nn.get_number_last_hidden_neurons()
    hidden_semantics = zeros((n_samples, n_neurons))
    for i, hidden_neuron in enumerate(nn.hidden_layers[-1]):
        hidden_semantics[:, i] = hidden_neuron.get_semantics()

    for output_index, output_neuron in enumerate(nn.output_layer):
        output_y = y[:, output_index]
        output_y_class_1_indices = where(output_y == 1)[0]
        output_y_class_1_count = output_y_class_1_indices.shape[0]
        output_y_class_0_count = n_samples - output_y_class_1_count

        sample_weights = ones(n_samples)
        class_1_weight = output_y_class_0_count / output_y_class_1_count
        sample_weights[output_y_class_1_indices] = class_1_weight

        reg = LinearRegression().fit(hidden_semantics, output_y, sample_weights)
        optimal_weights = np_append(reg.coef_.T, reg.intercept_)
        # ===================================================================
        # print('\n\toptimal_weights [min, mean, max]: [%.5f, %.5f, %.5f]' % (optimal_weights.min(), optimal_weights.mean(), optimal_weights.max()))
        # ===================================================================

        # Update connections with the learning step value:
        for i in range(n_neurons):
            output_neuron.input_connections[-n_neurons + i].weight = optimal_weights[i]

        output_neuron.increment_bias(optimal_weights[-1])


def init_lbfgs(X, y, nn, random_state):
    n_samples = y.shape[0]
    n_neurons = nn.get_number_last_hidden_neurons()
    hidden_semantics = zeros((n_samples, n_neurons))
    for i, hidden_neuron in enumerate(nn.hidden_layers[-1]):
        hidden_semantics[:, i] = hidden_neuron.get_semantics()

    layer_units = [n_neurons, y.shape[1]]
    # ===========================================================================
    # activations = [X]
    # ===========================================================================
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
        # output_neuron.increment_bias(intercepts[output_index])


# optimizer_function=init_lbfgs or optimizer_function=init_ols_balanced
def compute_weights(X, y, nn, random_state, optimizer_function=init_lbfgs):
    optimizer_function(X, y, nn, random_state)
