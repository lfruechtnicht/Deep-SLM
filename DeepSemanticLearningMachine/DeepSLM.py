from Hillclimbing.Hillclimbing import Hillclimbing
from DeepSemanticLearningMachine.NeuralNetwork import NeuralNetwork
import numpy as np
from algorithem.Metric import new_elite, CCE
import gc
from itertools import combinations_with_replacement, product


class DeepSLM(Hillclimbing):  # todo only for classification for now it should become abstract to also be for non image datasets
    """
    Parameters:
        layer_parameters: {'filters': list of Integers, the dimensionality of the output space (i.e. the number of
                                      output filters in the convolution).,
                           'kernel_sizes': list of integers or tuples/list of 2 integers, specifying the height and
                                           width of the 2D convolution window. Can be a single integer to specify the
                                           same value for all spatial dimensions.,
                           'strides': list of integer or tuple/list of 2 integers, specifying the strides of the
                                       convolution along the height and width. Can be a single integer to specify the
                                       same value for all spatial dimensions.,
                           'padding': list of  "valid" or "same" (case-insensitive).,
                           'activations': list of Activation functions to use. If None f(x)=x.,
                           'pool_size': list of integer or tuple of 2 integers, factors by which to downscale (vertical,
                                        horizontal). (2, 2) will halve the input in both spatial dimension. If only one 
                                        integer is specified, the same window length will be used for both dimensions.,
                           'neurons': list containing the number of neurons possible for the non convolutional part.}
    """

    def __init__(self,  # todo fix memory being used even tough not needed!
                 metric=CCE,
                 seed=None,  # todo add semantic stopping criterion
                 neighbourhood_size=10,  # todo add other mutations
                 stopping_criterion=None,  # todo add evolution of dense layers
                 max_depth_cl=20,  # todo add sigmoid evolution to las hidden layer
                 max_width_cl=1,
                 max_depth_non_conv=3,
                 max_width_non_conv=1,
                 max_splits=3,
                 layer_parameters=None):

        super().__init__(metric=metric,
                         seed=seed,
                         neighbourhood_size=neighbourhood_size,
                         stopping_criterion=stopping_criterion)

        self.max_depth_cl = max_depth_cl
        self.max_width_cl = max_width_cl
        self.max_depth_non_conv = max_depth_non_conv
        self.max_width_non_conv = max_width_non_conv
        self.max_splits = max_splits
        self.layer_parameters = layer_parameters
        self.conv_parameters, self.pool_parameters, self.non_conv_parameters = self._set_parameters()

    def fit(self, x_train, y_train, validation_data=None, verbose=False, validation_metric=None):
        if self.metric.type is "classification":
            try:
                n_outputs = y_train[0].shape[0]
            except IndexError:
                raise ValueError(f"If metric is set to classification targets of shape [n_samples, n_classes] expected!"
                                 f"Recieved: {y_train.shape}")
        elif self.metric.type is "regression":
            try:
                n_outputs = 1
            except IndexError:
                raise ValueError(f"If metric is set to regression targets of shape [n_samples] expected!"
                                 f"Recieved: {y_train.shape}")

        self._evolve(verbose, x_train, y_train, validation_data, n_outputs, validation_metric)

    def _evolve(self, verbose, x_train, y_train, validation_data, n_outputs, validation_metric):

        self.neighborhood = [NeuralNetwork(metric=self.metric,
                                           x_train=x_train,
                                           y_train=y_train,
                                           input_shape=x_train[0].shape,
                                           n_outputs=n_outputs,
                                           seed=self.random_state,
                                           layer_parameters=self.layer_parameters,
                                           conv_parameters=self.conv_parameters,
                                           pool_parameters=self.pool_parameters,
                                           non_conv_parameters=self.non_conv_parameters,
                                           max_depth_cl=self.max_depth_cl,
                                           max_width_cl=self.max_width_cl,
                                           max_depth_non_conv=self.max_depth_non_conv,
                                           max_width_non_conv=self.max_width_non_conv,
                                           max_splits=self.max_splits,
                                           validation_data=validation_data,
                                           validation_metric=validation_metric
                                           ) for _ in range(self.neighborhood_size)]
        self.neighborhood = sorted(self.neighborhood, key=lambda x: x.fitness, reverse=self.metric.greater_is_better)  # argmax
        self.elite = self.neighborhood[0]
        if verbose:
            self._verbose_reporter()
        del self.neighborhood
        gc.collect()

        for generation in range(100):
            self.current_generation += 1
            self.neighborhood = [self.elite.copy().isolated_mutation() for _ in
                                 range(self.neighborhood_size)]
            self.elite = self._get_elite()
            if verbose:
                self._verbose_reporter()
            del self.neighborhood
            gc.collect()

    def _set_parameters(self):
        filters, kernel_sizes, strides, padding, activations, pool_size, neurons = self.layer_parameters.values()
        _conv_parameters = [filters, kernel_sizes, strides, padding, activations]
        _pool_parameters = [pool_size, strides, padding]
        _full_parameters = [neurons, activations]
        return list(product(*_conv_parameters)), list(product(*_pool_parameters)), list(product(*_full_parameters))


if __name__ == '__main__':
    from tensorflow.keras.datasets import cifar10
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_classes = 10
    num_predictions = 20

    y_train = keras.utils.to_categorical(y_train, num_classes)
    classes = np.unique(y_train)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    x_train = x_train[:1000]
    y_train = y_train[:1000]

    DSLM = DeepSLM(seed=0)
    DSLM.fit(x_train, y_train, verbose=True)
    print(DSLM.predict(x_test).shape)

    # generate all solutions at the beginning for get_layer
