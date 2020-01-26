from tensorflow.keras.datasets import boston_housing
from tensorflow import keras
from DeepSemanticLearningMachine.DeepSLM import DeepSLM
from algorithem.Metric import *


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()



mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std

x_test -= mean
x_test /= std

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
filters = list(range(1, 10))
kernel_size = [(5, 5),
               (3, 3),
               (1, 1)
               ]
padding = ["valid",
           "same"]
pool_size = [(1, 1), (2, 2)]

neurons = list(range(1, 10))


layer_parameters = {"filters": filters,
                    "kernel_size": kernel_size,
                    "strides": strides,
                    "padding": padding,
                    "activations": activations,
                    "pool_size": pool_size,
                    "neurons": neurons}




DSLM = DeepSLM(CCE, seed=3, max_width_cl=1, neighbourhood_size=20, layer_parameters=layer_parameters)
DSLM.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=True)


"""WARNING: HIGH MEMORY REQUIREMENTS! TESTED ONLY WITH 16GB"""
