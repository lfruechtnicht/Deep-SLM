from tensorflow import keras
from tensorflow.keras import layers
from utils import check_random_state


class Node(object):
    """A node/layer in the directed graph which represents a Neural Network.
     Each node has exactly one input (except the input node) and at least one output node.
     """

    def __init__(self,
                 input_node=None,
                 is_input_node=False,
                 is_output_node=False,
                 computational_layer=None,
                 seed=None
                 ):

        self.input_node = input_node
        self.computational_layer = computational_layer
        self.is_input_node = is_input_node
        self.output_shape = None
        self.out_connections = []
        self.is_output_node = is_output_node  # TODO needs another evaluation that output node and output connection
        self.random_state = check_random_state(seed)
        self._evaluate()
        self._input_node_output_shape()
        if self.computational_layer is None:
            self._get_random_layer()
        self._connect_with_input()
        self._depth = 1  # depending on computational_layer is = to previous or plus one


    def _evaluate(self):
        """Evaluates the Node. Either it is an Input Node and has no Input Nodes or it has input nodes and is no Input
        Node. Moreover, a Input Node must be of type Node.
        Throws ValueError: A node can only be Input Node or have an Input Node! or ValueError: A Input Node must be a
        of type Node."""
        if not self.is_input_node and self.input_node is None:  # can also be list of nodes
            raise ValueError("A node can only be Input Node or have an Input Node or Input Nodes!")
        if not self.is_input_node and not isinstance(self.input_node, Node):
            if not isinstance(self.input_node, list) and not isinstance(self.computational_layer,
                                                                        keras.layers.concatenate):
                raise ValueError("""A Input Node can only be a list of Nodes for computational_layers of \n
                                     keras.layers.concatenate""")
            for node in self.input_node:
                if not isinstance(node, Node):
                    raise ValueError("Every element in the list must be of type Node!")
            raise ValueError("A Input Node must be a of type Node or a list with elements of type Node")

        if self.random_state is None:
            # todo get a radom state instacne
            pass

    def _input_node_output_shape(self):
        """sets the the output shape of an input node."""
        if self.is_input_node:
            self.output_shape = self.computational_layer.shape

    def _connect_with_input(self):
        """Connects the current node with the input node. This means setting the input of the computational_layer,
        setting a output connection for the input node and determining a output shape of the current node."""
        if not self.is_input_node:
            self.input_node.out_connections.append(self)
            self.computational_layer = self.computational_layer(self.input_node.computational_layer)
            self.output_shape = self.computational_layer.shape
            if self.computational_layer == keras.layers.concatenate:
                self._depth = self.input_node._depth
            elif not self.computational_layer == keras.layers.Input:
                self._depth = self.input_node._depth + 1

    def _get_random_layer(self):
        pass

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


        _layer = self.random_state.choice([0])
        if _layer == 0:
            strides = [(2, 2), (1, 1)]
            _filters = list(range(50))
            kernel_size = (5,5)
            filters = self.random_state.choice(_filters)

            strides = strides[self.random_state.choice(len(strides))]
            padding = self.random_state.choice(["valid", "same"])
            activation = self.random_state.choice(activations)
            layer = keras.layers.Conv2D(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        activation=activation)





        else:
            _pooling = self.random_state.choice([0,1])
            pool_size = (2, 2)
            strides = [(2, 2), (1, 1)]
            strides = strides[self.random_state.choice(len(strides))]
            padding = 'valid'

            if _pooling == 0:
                layer = keras.layers.AveragePooling2D(pool_size=pool_size,
                                                  strides=strides,
                                                  padding=padding)
            else:
                layer = keras.layers.MaxPool2D(pool_size=pool_size,
                                                  strides=strides,
                                                  padding=padding)
        self.computational_layer = layer





if __name__ == '__main__':
    B = 1
    # A = Node(input_node=B)
    C = Node(is_input_node=True, computational_layer=keras.Input(shape=(32, 32, 3)))
    D = Node(input_node=C, computational_layer=layers.Conv2D(64, 3, activation='relu', padding='same'))

    E = Node(input_node=C, computational_layer=keras.layers.Convolution2D(
       filters=(65),kernel_size=(5,5), padding='same', activation='relu'))
    F = Node(input_node=C,seed=0)

    print(D.output_shape.ndims > 2)
    print(C.output_shape)
    print(D.computational_layer)
