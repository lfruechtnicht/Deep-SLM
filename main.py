from tensorflow.keras.datasets import cifar10
from tensorflow import keras
from DeepSemanticLearningMachine.DeepSLM import DeepSLM

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10
num_predictions = 20

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

DSLM = DeepSLM(seed=1)
DSLM.fit(x_train, y_train, verbose=True)


"""WARNING: HIGH MEMORY REQUIREMENTS! TESTED ONLY WITH 16GB"""
