from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

input_shape = (28,28,1)

activation = 'softsign'

#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, input_shape=input_shape))
model.add(Activation(activation))
