from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

input_shape = (28, 28, 1)

activation = 'softsign'  # ... all are possible ...
filters = 4
kernel_size = (2, 2)
stride = (1, 1)
padding = "same"
# create model
model = Sequential()

# add model layers
model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, input_shape=input_shape))
model.add(Activation(activation))
