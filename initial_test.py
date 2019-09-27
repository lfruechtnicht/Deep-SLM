from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import itertools


from utils import *
input_shape = (28, 28, 1)


activations = ["softmax",
               "elu",
               "selu",
               "softplus",
               "softsign",
               "relu",
               "tanh",
               "sigmoid",
               "hard_sigmoid",
               "exponential",
               "linear",
               ]
filters = list(range(1,100))
kernel_size = (2, 2)
stride = (1, 1)
padding = "same"
# create model


seed = 4

populationsize = 2
random_state = check_random_state(seed)

seeds = random_state.random_integers(1000000, size=populationsize)

population = []
for i in range(populationsize):

    print("hello")
    print(i)

    #

    model = Sequential()


    # build intial CL
    # generate randome states for each parameter
    random_state = check_random_state(seeds[i])
    _random_seeds = random_state.random_integers(1000000, size=5)
    # determine filter size
    filters = list(range(1, 100))
    random_state = check_random_state(_random_seeds[0])
    _filter = random_state.choice(filters)
    # choose kernel_size
    _kernel_ = (1, 2,3,4,5,6,7)
    kernel_sizes = list(itertools.product(_kernel_, _kernel_))
    random_state = check_random_state(_random_seeds[1])
    _kernel = kernel_sizes[random_state.choice(len(kernel_sizes),1)[0]]
    # choose padding


    model.add(Conv2D(filters=_filter, kernel_size=_kernel, padding=padd, strides=stride, input_shape=input_shape))


    random_state = check_random_state(_random_seeds[4])
    __activation = random_state.choice(activations)
    #print(__activation)
    model.add(Activation(__activation))

    population.append(model)

    neurons_last_layer = np.prod(model.layers[-1].output_shape[1:])
print(population)