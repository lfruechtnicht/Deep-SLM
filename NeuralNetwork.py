from tensorflow import keras
from tensorflow.keras import layers
from Node import Node


class NeuralNetwork(object):
    """ A neural network object consists of  an input Nodes, intermediate Nodes and an output Node. Together they form
    the model which is a Keras computational model."""

    def __init__(self, max_depth_cl, max_depth_ff, input_shape):
        self.max_depth_cl = max_depth_cl
        self.max_depth_ff = max_depth_ff
        self.input_shape = input_shape
        self.input_node = None
        self.inter_nodes = []
        self.output_node = None
        self.model = None
        self.semantics = None

    def build_cl(self):
        # 1 define the input node
        self.input_node = Node(is_input_node=True, computational_layer=keras.Input(shape=self.input_shape))

    def build_ff(self):
        pass
