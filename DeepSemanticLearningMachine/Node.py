import sys
from tensorflow import keras
from utils import check_random_state
import numpy as np

import uuid



class Node(object):
    """A node/layer in the directed graph which represents a Neural Network.
     Attribute:
     mutation_level: Int The number of mutation which is currently performed
     conv_parameters: list containing all possible combinations of parameters for keras conv layers
     pool_parameters: list containing all possible combinations of parameters for keras pool layers
     is_input_node: Bool If this node is the initial input exposed to the raw data
     _computational_layer tf.keras.layers Layer A reusable layer or none
     cl Bool if this is a node of the convolutional part of the network (not used yet)
     seed Int or RandomState instance
     input_shape Tuple Specifies the input_shape for the input node
     semantics np.array the saved output of the semantics_computational_layer
     input_data np.array the semantics of the connected node
     semantic_input Bool if the node receives direct input from semantics of a previous node
     semantic_input_node keras.Input helper input to handle the computation for the semantics_computational_layer
     computational_layer tf.keras.layer layer that is connect to the input of the network and the output of the network
     semantics_computational_layer tf.keras.layer that only acts as a dummy to calculate the semantics
     input_node Node or list of Nodes which this node will be connected to
     depth Int depth in the network
     """

    def __init__(self,
                 mutation_level,
                 conv_parameters=None,
                 pool_parameters=None,
                 non_conv_parameters=None,
                 is_input_node=False,
                 _computational_layer=None,
                 layer_type="conv",
                 seed=None,
                 input_shape=None,
                 semantics=None,
                 input_data=None,
                 semantic_input=False,
                 semantic_input_node=None,
                 computational_layer=None,
                 semantics_computational_layer=None,
                 input_node=None,
                 depth=1
                 ):

        self.mutation_level = mutation_level
        self.conv_parameters = conv_parameters
        self.pool_parameters = pool_parameters
        self.non_conv_parameters = non_conv_parameters
        self.input_node = input_node
        self._computational_layer = _computational_layer
        self.computational_layer = computational_layer
        self.is_input_node = is_input_node
        self.layer_type = layer_type
        self.input_shape = input_shape
        self.output_shape = None
        self.out_connections = []
        self.depth = depth  # depending on computational_layer is = to previous or plus one
        self.random_state = check_random_state(seed)
        self.semantics = semantics
        self.semantic_input = semantic_input
        self.semantic_input_node = semantic_input_node
        self.input_data = input_data
        self.semantics_computational_layer = semantics_computational_layer
        if self._computational_layer is None:
            self._get_random_layer()
        if semantics is None:
            self.set_computational_layer()
        self._input_node_output_shape()

    def __repr__(self):
        return str(self.computational_layer)

    def __copy__(self):
        is_input_node = self.is_input_node
        layer_type = self.layer_type
        computational_layer = self.computational_layer
        random_state = self.random_state
        depth = self.depth
        semantics = self.semantics
        mutation_level = self.mutation_level
        semantic_input = self.semantic_input
        semantic_input_node = self.semantic_input_node
        input_data = self.input_data
        semantics_computational_layer = self.semantics_computational_layer

        return Node(is_input_node=is_input_node,
                    layer_type=layer_type,
                    computational_layer=computational_layer,
                    seed=random_state,
                    depth=depth,
                    semantics=semantics,
                    mutation_level=mutation_level,
                    semantic_input=semantic_input,
                    semantic_input_node=semantic_input_node,
                    input_data=input_data,
                    semantics_computational_layer=semantics_computational_layer)

    def _input_node_output_shape(self):
        """sets the the output shape of an input node."""
        if self.is_input_node:
            self.output_shape = self.computational_layer.shape.dims

    def set_computational_layer(self):
        """Connects the current node with it's input node(s). This means setting the input of the computational_layer,
        setting a output connection for the input node and determining a output shape of the current node.
        Also takes care of connecting the semantics graph"""

        if isinstance(self.input_node, list):  # connect for concatenation or add
            input_nodes_merge_layer = [node.computational_layer for node in self.input_node]  # get all nodes to connect

            if self.mutation_level == 0:  # connect initial network, no special semantics graph needed
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
                raise ValueError("also something is wrong")

    def connect_main_layers(self):
        """Connects the layers of the main graph"""
        self.computational_layer = self._computational_layer(self.input_node.computational_layer)
        self.output_shape = self.computational_layer.shape.dims
        self.input_node.out_connections.append(self)
        self.depth = self.input_node.depth + 1

    def connect_merge_layers(self, input_nodes_merge_layer):
        """connects the merge layers of the main graph"""
        self.computational_layer = self._computational_layer(input_nodes_merge_layer)
        self.output_shape = self.computational_layer.shape.dims
        _depths = []
        for input_node in self.input_node:
            input_node.out_connections.append(self)
            _depths.append(input_node.depth)
        self.output_shape = self.computational_layer.shape.dims
        self.depth = max(_depths)

    def only_flatten_test(self):
        """Makes sure that dimensions don't get negative"""
        if self.output_shape[1] < 5 or self.output_shape[2] < 5:
            out = True
        else:
            out = False
        return out

    def _get_random_layer(self):
        """Chooses a keras layer of possible choices"""

        _seed = self.random_state.randint(sys.maxsize)  # to be able to reproduce results
        kernel_initializer = self.random_state.choice([keras.initializers.RandomNormal(seed=_seed),
                                                       keras.initializers.RandomUniform(seed=_seed)])

        if self.layer_type == "conv":
            _layer = self.random_state.choice([0, 1])  # choose Conv or Pooling, can be extended
            if _layer == 0:
                idx = self.random_state.choice(len(self.conv_parameters))
                _filters = self.conv_parameters[idx][0]
                _kernel_size = self.conv_parameters[idx][1]
                _strides = self.conv_parameters[idx][2]
                _padding = self.conv_parameters[idx][3]
                _activation = self.conv_parameters[idx][4]
                layer = keras.layers.Conv2D(filters=_filters,
                                            kernel_size=_kernel_size,
                                            strides=_strides,
                                            padding=_padding,
                                            activation=_activation,
                                            kernel_initializer=kernel_initializer)
            else:
                _pooling = self.random_state.choice([0, 1])
                idx = self.random_state.choice(len(self.pool_parameters))
                _pool_size = self.pool_parameters[idx][0]
                _strides = self.pool_parameters[idx][1]
                _padding = self.pool_parameters[idx][2]

                if _pooling == 0:
                    layer = keras.layers.AveragePooling2D(pool_size=_pool_size,
                                                          strides=_strides,
                                                          padding=_padding)
                else:
                    layer = keras.layers.MaxPool2D(pool_size=_pool_size,
                                                   strides=_strides,
                                                   padding=_padding)
        else:
            idx = self.random_state.choice(len(self.non_conv_parameters))
            _neurons = self.non_conv_parameters[idx][0]
            _activations = self.non_conv_parameters[idx][1]
            layer = keras.layers.Dense(units=_neurons, activation=_activations, kernel_initializer=kernel_initializer)

        self._computational_layer = layer


if __name__ == '__main__':
    B = 1
    # A = Node(input_node=B)
    X = Node(is_input_node=True, computational_layer=keras.Input(shape=(5, 5, 3)))
    print(X.only_flatten_test())
    X = Node(input_node=X, computational_layer=None, seed=41)
    print(X.only_flatten_test())
    # X = Node(input_node=X, computational_layer=None, seed=32)
    print(X.only_flatten_test())
