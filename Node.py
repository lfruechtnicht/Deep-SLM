import sys
from tensorflow import keras
from utils import check_random_state
import uuid

# layer parameters

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
strides = [(3, 3)#,
           #(2, 2),
           #(1, 1)
           ]
filters = list(range(50))
kernel_size = [(5, 5)#,
               #(3, 3),
               #(1, 1)
               ]
padding = ["valid",
           "same"]
pool_size = [(1, 1), (2, 2)]


class Node(object):
    """A node/layer in the directed graph which represents a Neural Network.
     Each node has exactly one input (except the input node) and at least one output node.
     """

    def __init__(self,
                 initial_input_node,
                 tmpdir,
                 input_node=None,
                 is_input_node=False,
                 computational_layer=None,
                 cl=True,
                 seed=None
                 ):

        self.root_tmpdir = tmpdir
        self.initial_input_node = initial_input_node
        self.input_node = input_node
        self.computational_layer = computational_layer
        self.is_input_node = is_input_node
        self.cl = cl
        self.output_shape = None
        self.out_connections = []
        self.depth = 1  # depending on computational_layer is = to previous or plus one
        self.random_state = check_random_state(seed)
        self._eval()
        if not self.is_input_node:
            self._tmpdir = # todo
        self._input_node_output_shape()
        if self.computational_layer is None:
            self._get_random_layer()
        self._connect_with_input()
        self.model = None
        self.semantics = None

    def _eval(self):
        """Evaluates the Node. Either it is an Input Node and has no Input Nodes or it has input nodes and is no Input
        Node. Moreover, a Input Node must be of type Node.
        Throws ValueError: A node can only be Input Node or have an Input Node! or ValueError: A Input Node must be a
        of type Node."""
        if not self.is_input_node and self.input_node is None:  # can also be list of nodes
            raise ValueError("A node can only be Input Node or have an Input Node or Input Nodes!")
        if not self.is_input_node and not isinstance(self.input_node, Node):
            if not isinstance(self.input_node, list):
                raise ValueError("""A Input Node can only be a list of Nodes for computational_layers of \n
                                     keras.layers.concatenate""")

            for node in self.input_node:
                if not isinstance(node, Node):
                    raise ValueError("A Input Node must be a of type Node or a list with elements of type Node")

    def _input_node_output_shape(self):
        """sets the the output shape of an input node."""
        if self.is_input_node:
            self.output_shape = self.computational_layer.shape

    def _connect_with_input(self):
        """Connects the current node with the input node. This means setting the input of the computational_layer,
        setting a output connection for the input node and determining a output shape of the current node."""
        if not self.is_input_node and not isinstance(self.input_node, list):
            self.input_node.out_connections.append(self)
            self.computational_layer = self.computational_layer(self.input_node.computational_layer)
            self.output_shape = self.computational_layer.shape
            if not self.computational_layer == keras.layers.Input:
                self.depth = self.input_node.depth + 1
        if not self.is_input_node and isinstance(self.input_node, list):
            _depths = []
            for input_node in self.input_node:
                input_node.out_connections.append(self)
                _depths.append(input_node.depth)
            self.output_shape = self.computational_layer.shape
            self.depth = max(_depths)

    def only_flatten_test(self):
        if self.output_shape[1] < 5 or self.output_shape[2] < 5:
            out = True
        else:
            out = False
        return out

    def _get_random_layer(self):

        _seed = self.random_state.randint(sys.maxsize)
        kernel_initializer = self.random_state.choice([keras.initializers.RandomNormal(seed=_seed),
                                                       keras.initializers.RandomUniform(seed=_seed)])

        if self.cl:
            # todo irandom state instance test! This sysmax is absurd!
            _layer = self.random_state.choice([0, 1])
            _strides = strides[self.random_state.choice(len(strides))]
            _padding = self.random_state.choice(padding)

            if _layer == 0:

                _filters = self.random_state.choice(filters)
                _activation = self.random_state.choice(activations)
                _kernel_size = kernel_size[self.random_state.choice(len(kernel_size))]
                layer = keras.layers.Conv2D(filters=_filters,
                                            kernel_size=_kernel_size,
                                            strides=_strides,
                                            padding=_padding,
                                            activation=_activation,
                                            kernel_initializer=kernel_initializer)
            else:
                _pooling = self.random_state.choice([0, 1])
                _pool_size = pool_size[self.random_state.choice(len(pool_size))]

                if _pooling == 0:
                    layer = keras.layers.AveragePooling2D(pool_size=_pool_size,
                                                          strides=_strides,
                                                          padding=_padding)
                else:
                    layer = keras.layers.MaxPool2D(pool_size=_pool_size,
                                                   strides=_strides,
                                                   padding=_padding)
            self.computational_layer = layer

    def _get_intermediate_model(self):
        if not self.is_input_node:
            self.model = keras.models.Model(inputs=self.initial_input_node.computational_layer,
                                            outputs=self.computational_layer)
        else:
            pass

    def get_semantics(self, data):
        if not self.is_input_node:
            self._get_intermediate_model()
            self.semantics = self.model.predict(data)
            del self.model
        else:
            self.semantics = data









if __name__ == '__main__':
    B = 1
    # A = Node(input_node=B)
    X = Node(is_input_node=True, computational_layer=keras.Input(shape=(5, 5, 3)))
    print(X.only_flatten_test())
    X = Node(input_node=X, computational_layer=None, seed=41)
    print(X.only_flatten_test())
    #X = Node(input_node=X, computational_layer=None, seed=32)
    print(X.only_flatten_test())


