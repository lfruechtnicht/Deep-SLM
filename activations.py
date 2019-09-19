import keras
from keras.layers import Activation
import random
from utils import *


random_state = check_random_state(0)

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
activation = random_state.choice(activations)

print(activation)