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
strides = [(3, 3)  # ,
           # (2, 2),
           # (1, 1)
           ]
filters = list(range(50))
kernel_size = [(5, 5)  # ,
               # (3, 3),
               # (1, 1)
               ]
padding = ["valid",
           "same"]
pool_size = [(1, 1), (2, 2)]


# todo initilaize all once that only one randome chioce has to be made!

class Node(object):
    """A node/layer in the directed graph which represents a Neural Network.
     Each node has exactly one input (except the input node) and at least one output node.
     """

    def __init__(self,
                 mutation_level,
                 input_node=None,
                 is_input_node=False,
                 computational_layer=None,
                 cl=True,
                 seed=None,
                 input_shape=None
                 ):
        self.mutation_level = mutation_level
        self.input_node = input_node
        self._computational_layer = computational_layer
        self.computational_layer = None
        self.is_input_node = is_input_node
        self.cl = cl
        self.input_shape = input_shape
        self.output_shape = None
        self.out_connections = []
        self.depth = 1  # depending on computational_layer is = to previous or plus one
        self.random_state = check_random_state(seed)
        self._eval()
        self.semantic_input = False
        self.semantic_input_node = None
        self.input_data = None
        self.semantics_computational_layer = None
        if self._computational_layer is None:
            self._get_random_layer()
        self.set_computational_layer()
        self._input_node_output_shape()

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
            self.output_shape = self.computational_layer.shape.dims

    def set_computational_layer(self):
        """Connects the current node with it's input node. This means setting the input of the computational_layer,
        setting a output connection for the input node and determining a output shape of the current node."""

        if isinstance(self.input_node, list):  # connect for concatenation or add
            input_nodes_merge_layer = [node.computational_layer for node in self.input_node]  # get all nodes to connect

            if self.mutation_level == 0:  # connect initial network
                self.connect_merge_layers(input_nodes_merge_layer)

            else:  # connect also semantic layers
                mutation_levels = [node.mutation_level for node in self.input_node]
                # connect only nodes from one mutation level without new semantic input
                if len(set(mutation_levels)) == 1 and mutation_levels[0] == self.mutation_level:
                    # get all nodes to connect
                    input_nodes_merge_layer_semantics = [node.semantics_computational_layer for node in self.input_node]
                    self.semantics_computational_layer = self._computational_layer(input_nodes_merge_layer_semantics)
                    self.connect_merge_layers(input_nodes_merge_layer)

                # connect nodes from different mutation level with new semantic input
                elif len(set(mutation_levels)) != 1:
                    self.semantic_input = True  # to filter for input nodes
                    # get all nodes with new semantic input and other:
                    no_new_semantic_input = [node.semantics_computational_layer for node in self.input_node if
                                             node.mutation_level == self.mutation_level]  # fix  this
                    new_semantic_input_node = [node for node in self.input_node if  # to easily find the data
                                               node.semantics_computational_layer not in no_new_semantic_input]

                    self.input_data = [node.semantics for node in new_semantic_input_node]

                    # Input for semantics are the new inputs AND the non new inputs if applicable
                    self.semantic_input_node = [keras.Input(shape=node.output_shape[1:]) for node
                                                in new_semantic_input_node]

                    self.semantics_computational_layer = self._computational_layer(self.semantic_input_node +
                                                                                   no_new_semantic_input)
                    self.connect_merge_layers(input_nodes_merge_layer)

                else:
                    raise ValueError("Something is wrong!")

        else:  # connect normal layers
            if self.mutation_level == 0:  # connect initial network

                if self.is_input_node:  # initialize the initial input which can only happen once!
                    self.computational_layer = self._computational_layer
                else:  # set any other computational layer
                    self.connect_main_layers()

            elif self.mutation_level == self.input_node.mutation_level:  # connect normal mut layer WITH semantics!
                self.semantics_computational_layer = \
                    self._computational_layer(self.input_node.semantics_computational_layer)

                self.connect_main_layers()

            elif self.mutation_level > self.input_node.mutation_level:  # connect mut layer WITH new semantic input!
                self.semantic_input = True  # to filter for input nodes
                self.input_data = self.input_node.semantics  # to easily find the data
                self.semantic_input_node = keras.Input(shape=self.input_node.output_shape[1:])  # Input for semantics
                self.semantics_computational_layer = self._computational_layer(self.semantic_input_node)

                self.connect_main_layers()
            else:
                raise ValueError("also somethign is wrong")

    def connect_main_layers(self):
        self.computational_layer = self._computational_layer(self.input_node.computational_layer)
        self.output_shape = self.computational_layer.shape.dims
        self.input_node.out_connections.append(self)
        self.depth = self.input_node.depth + 1

    def connect_merge_layers(self, input_nodes_merge_layer):
        self.computational_layer = self._computational_layer(input_nodes_merge_layer)
        self.output_shape = self.computational_layer.shape.dims
        _depths = []
        for input_node in self.input_node:
            input_node.out_connections.append(self)
            _depths.append(input_node.depth)
        self.output_shape = self.computational_layer.shape.dims
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
            # todo i random state instance test! This sysmax is absurd!
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
            self._computational_layer = layer

    # def get_semantics(self, data):
    #     if self.is_input_node:
    #         self.semantics = data
    #     elif self.semantics is None:
    #         if isinstance(self.input_node, Node):
    #             semantics_input = keras.Input(shape=self.input_node.output_shape[1:])
    #             semantics_computational_layer = self._semantics_computational_layer(semantics_input)
    #
    #             self.model = keras.models.Model(inputs=semantics_input,
    #                                             outputs=semantics_computational_layer)
    #
    #             self.semantics = self.model.predict(self.input_node.semantics)
    #             del self.model, self._semantics_computational_layer
    #
    #         if isinstance(self.input_node, list):
    #
    #             semantics_input = [keras.Input(shape=node.output_shape[1:]) for node in self.input_node]
    #
    #             if self._computational_layer is "add":
    #                 semantics_computational_layer = keras.layers.add(inputs=semantics_input)
    #             elif self._computational_layer is "concatenate":
    #                 semantics_computational_layer = keras.layers.concatenate(inputs=semantics_input)
    #
    #             self.model = keras.models.Model(inputs=semantics_input,
    #                                             outputs=semantics_computational_layer)
    #
    #             input_semantics = [node.semantics for node in self.input_node]
    #             self.semantics = self.model.predict(input_semantics)
    #             del self.model


if __name__ == '__main__':
    B = 1
    # A = Node(input_node=B)
    X = Node(is_input_node=True, computational_layer=keras.Input(shape=(5, 5, 3)))
    print(X.only_flatten_test())
    X = Node(input_node=X, computational_layer=None, seed=41)
    print(X.only_flatten_test())
    # X = Node(input_node=X, computational_layer=None, seed=32)
    print(X.only_flatten_test())
