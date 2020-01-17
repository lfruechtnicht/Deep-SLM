from Hillclimbing.Hillclimbing import Hillclimbing
from DeepSemanticLearningMachine import NeuralNetwork
from utils import check_random_state
import numpy as np
from copy import copy
from algorithem.Metric import new_elite


class DeepSLM(Hillclimbing):  # only for classification for now it should become abstract

    def __init__(self,neighbourhood_size=10, seed=None, stopping_criterion=None):
        super().__init__(neighbourhood_size, seed, stopping_criterion)

    def fit(self, x_train, y_train, metric=None, verbose=False):
        self.elite = NeuralNetwork(x_train=x_train,
                                   y_train=y_train,
                                   input_shape=x_train[0].shape,
                                   n_outputs=y_train[0].shape[0],
                                   seed=self.seed
                                   )
        self._evolve(verbose)

    def _evolve(self, verbose):
        for generation in range(10):
            self.neighborhood = [self.elite.copy().single_mutation() for _ in range(10)]
            self.neighborhood.sort(key=lambda x: x.fitness, reverse=False)
            self.elite = new_elite(self.neighborhood[0], self.elite)
            if verbose:
                print(f"Generation: {self.current_generation}, Fitness: {self.elite.fitness:.5f}")
            del self.neighborhood
            self.current_generation += 1



if __name__ == '__main__':
    from DeepSemanticLearningMachine.NeuralNetwork import NeuralNetwork
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

    DSLM = DeepSLM(seed=0)
    DSLM.fit(x_train, y_train, verbose=True)
    print(DSLM.predict(x_test).shape)

    #todo documentation
    # check beginning of mutation
    # check only one RNG
    # generate all solutions at the beginning for get_layer


