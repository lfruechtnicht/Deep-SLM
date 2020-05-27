from tensorflow.keras.datasets import cifar10
from tensorflow import keras
from DeepSemanticLearningMachine.DeepSLM import DeepSLM
from algorithem.Metric import *


# todo: Building the network, espeicially the cp!
# todo: seperate the initialization and mutation of the network!
# todo: Test mutation af a function to be be applied in parralel!
# todo: build mutation in practice!


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10
num_predictions = 20

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

activations = [
    "elu",
    "selu",

    "relu",
    "tanh",
    "sigmoid",
    "hard_sigmoid",
    "exponential",
    "linear",
    ]
strides = [(3, 3),
           (2, 2),
           (1, 1)
           ]
filters = list(range(1, 3))
kernel_size = [(5, 5),
               (3, 3),
               (1, 1)
               ]
padding = ["valid",
           "same"]
pool_size = [(1, 1), (2, 2)]

neurons = list(range(10, 20))

layer_parameters = {"filters": filters,
                    "kernel_size": kernel_size,
                    "strides": strides,
                    "padding": padding,
                    "activations": activations,
                    "pool_size": pool_size,
                    "neurons": neurons}

x_train = x_train[:1000]
y_train = y_train[:1000]

""" TODO: -Lukas main.py """
DSLM = DeepSLM(RootMeanSquaredError, seed=4, max_depth_cl=10, max_width_cl=1, max_depth_non_conv=4,
               max_width_non_conv=3, neighbourhood_size=1, layer_parameters=layer_parameters)
#===============================================================================
# DSLM = DeepSLM(RootMeanSquaredError, seed=1, max_depth_cl=10, max_width_cl=1, max_depth_non_conv=4,
#                max_width_non_conv=3, neighbourhood_size=25, layer_parameters=layer_parameters)
#===============================================================================
DSLM.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=True, validation_metric=Accurarcy)  # todo validation true false



"""WARNING: HIGH MEMORY REQUIREMENTS! TESTED ONLY WITH 16GB"""
