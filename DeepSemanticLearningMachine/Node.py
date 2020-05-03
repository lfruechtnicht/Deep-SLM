import sys
from tensorflow import keras


class Node(object):
    """
    Abstract class for Node objects in NeuralNetwork.
    """
    def __init__(self,
                 mutation_level,
                 input_node=None,
                 depth=1):

        self.mutation_level = mutation_level
        self.input_node = input_node
        self.depth = depth  # depending on computational_layer is = to previous or plus one

        self.output_shape = None
        self.out_connections = []
        self.semantics = None
        self._computational_layer = None  # the layer to be shared
        self.computational_layer = None  # the layer connected to the main graph
        self.input_data = None
        self.semantic_input = False
        self.semantic_input_node = None
        self.semantics_computational_layer = None  # the layer connected to the semantics graph

    def __repr__(self):
        return str(self._computational_layer.name)

    def _set_computational_layer(self):
        """
        Connects the relevant uses the relevant _computational_layer to connect the node to the main computational
        graph of the network and also to connect the node to a mutation_level specific computational graph which is does
        not take the original data as an input but previously calculated semantics. This semantics graph has the sole
        purpose to calculate the effect of the new mutation on the whole network.
        :return: void
        """
        pass


class InputNode(Node):
    def __init__(self,
                 mutation_level,
                 input_shape,
                 input_node=None):
        super().__init__(mutation_level, input_node)
        self.computational_layer = keras.Input(shape=input_shape)
        self.output_shape = self.computational_layer.shape.dims

    def only_flatten_test(self):
        """Makes sure that dimensions don't get negative"""
        if self.output_shape[1] < 5 or self.output_shape[2] < 5:
            out = True
        else:
            out = False
        return out


class FlattenNode(Node):
    def __init__(self, mutation_level, input_node):
        super().__init__(mutation_level, input_node)
        self._computational_layer = keras.layers.Flatten()
        self._set_computational_layer()

    def _connect_main_layers(self, depth=False):
        """Connects the layers of the main graph"""
        self.computational_layer = self._computational_layer(self.input_node.computational_layer)
        self.output_shape = self.computational_layer.shape.dims
        self.input_node.out_connections.append(self)
        if depth:
            self.depth = 1
        else:
            self.depth = self.input_node.depth + 1

    def _set_computational_layer(self):
        if self.mutation_level == 0:  # connect initial network
            self._connect_main_layers()

        elif self.mutation_level == self.input_node.mutation_level:  # connect normal mutation layer WITH semantics!
            self.semantics_computational_layer = self._computational_layer(
                self.input_node.semantics_computational_layer)
            self._connect_main_layers()

        elif self.mutation_level > self.input_node.mutation_level:  # connect mutation layer WITH new semantic input!
            self.semantic_input = True  # to filter for input nodes
            self.input_data = self.input_node.semantics  # to easily find the data
            self.semantic_input_node = keras.Input(shape=self.input_node.output_shape[1:])  # Input for semantics
            self.semantics_computational_layer = self._computational_layer(self.semantic_input_node)
            self._connect_main_layers(depth=True)
        else:
            raise ValueError("also something is wrong")


class ConvNode(Node):
    def __init__(self, mutation_level, input_node, random_state, conv_parameters, pool_parameters):
        super().__init__(mutation_level, input_node)
        self.random_state = random_state
        self.conv_parameters = conv_parameters
        self.pool_parameters = pool_parameters
        self._get_random_layer()
        self._set_computational_layer()

    def _get_random_layer(self):
        """Chooses a keras layer of possible choices"""

        _seed = self.random_state.randint(1E6)  # to be able to reproduce results
        kernel_initializer = self.random_state.choice([keras.initializers.RandomNormal(seed=_seed),
                                                       keras.initializers.RandomUniform(seed=_seed)])

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
        self._computational_layer = layer

    def only_flatten_test(self):
        """Makes sure that dimensions don't get negative"""
        if self.output_shape[1] < 5 or self.output_shape[2] < 5:
            out = True
        else:
            out = False
        return out

    def _connect_main_layers(self, depth=False):
        """Connects the layers of the main graph"""
        self.computational_layer = self._computational_layer(self.input_node.computational_layer)
        self.output_shape = self.computational_layer.shape.dims
        self.input_node.out_connections.append(self)
        if depth:
            self.depth = 1
        else:
            self.depth = self.input_node.depth + 1

    def _set_computational_layer(self):
        if self.mutation_level == 0:  # connect initial network
            self._connect_main_layers()

        elif self.mutation_level == self.input_node.mutation_level:  # connect normal mut layer WITH semantics!
            self.semantics_computational_layer = self._computational_layer(
                self.input_node.semantics_computational_layer)
            self._connect_main_layers()

        elif self.mutation_level > self.input_node.mutation_level:  # connect mut layer WITH new semantic input!
            self.semantic_input = True  # to filter for input nodes
            self.input_data = self.input_node.semantics  # to easily find the data
            self.semantic_input_node = keras.Input(shape=self.input_node.output_shape[1:])  # Input for semantics
            self.semantics_computational_layer = self._computational_layer(self.semantic_input_node)
            self._connect_main_layers(depth=True)
        else:
            raise ValueError("also something is wrong")


class MergeNode(Node):
    def __init__(self, mutation_level, input_node, _computational_layer):
        super().__init__(mutation_level, input_node)
        self._computational_layer = _computational_layer
        self._set_computational_layer()

    def _connect_merge_layers(self, input_nodes):
        """connects the merge layers of the main graph"""
        self.computational_layer = self._computational_layer(input_nodes)
        self.output_shape = self.computational_layer.shape.dims
        _depths = []
        for input_node in self.input_node:
            input_node.out_connections.append(self)
            _depths.append(input_node.depth)
        self.depth = min(_depths)

    def _set_computational_layer(self):
        input_nodes = [node.computational_layer for node in self.input_node]  # get all nodes to connect
        if self.mutation_level == 0:  # connect initial network, no special semantics graph needed
            self._connect_merge_layers(input_nodes)

        else:  # connect also semantic layers
            mutation_levels = [node.mutation_level for node in self.input_node]
            # connect only nodes from one mutation level without new semantic input
            if len(set(mutation_levels)) == 1 and mutation_levels[0] == self.mutation_level:
                # get all nodes to connect
                input_nodes_merge_layer_semantics = [node.semantics_computational_layer for node in self.input_node]
                self.semantics_computational_layer = self._computational_layer(input_nodes_merge_layer_semantics)
                self._connect_merge_layers(input_nodes)

            # connect nodes from different mutation level with new semantic input
            # elif len(set(mutation_levels)) != 1:
            else:
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
                self._connect_merge_layers(input_nodes)
            #
            # else:
            #     raise ValueError("Something is wrong!")

    def only_flatten_test(self):
        """Makes sure that dimensions don't get negative"""
        if self.output_shape[1] < 5 or self.output_shape[2] < 5:
            out = True
        else:
            out = False
        return out


class DenseNode(Node):
    def __init__(self, mutation_level, input_node, random_state, non_conv_parameters=None, _computational_layer=None):
        super().__init__(mutation_level, input_node)
        self.random_state = random_state
        self.non_conv_parameters = non_conv_parameters
        self._computational_layer = _computational_layer
        self._set_computational_layer()

    def _connect_main_layers(self):
        """Connects the layers of the main graph"""
        self.computational_layer = self._computational_layer(self.input_node.computational_layer)
        self.output_shape = self.computational_layer.shape.dims
        self.input_node.out_connections.append(self)
        self.depth = self.input_node.depth + 1

    def _set_computational_layer(self):
        if self.mutation_level == 0:  # connect initial network
            self._connect_main_layers()

        elif self.mutation_level == self.input_node.mutation_level:  # connect normal mut layer WITH semantics!
            self.semantics_computational_layer = self._computational_layer(
                self.input_node.semantics_computational_layer)
            self._connect_main_layers()

        elif self.mutation_level > self.input_node.mutation_level:  # connect mut layer WITH new semantic input!
            self.semantic_input = True  # to filter for input nodes
            self.input_data = self.input_node.semantics  # to easily find the data
            self.semantic_input_node = keras.Input(shape=self.input_node.output_shape[1:])  # Input for semantics
            self.semantics_computational_layer = self._computational_layer(self.semantic_input_node)
            self._connect_main_layers()
        else:
            raise ValueError("also something is wrong")


    def _get_random_layer(self):
        _seed = self.random_state.randint(1E7)  # to be able to reproduce results
        kernel_initializer = self.random_state.choice([keras.initializers.RandomNormal(seed=_seed),
                                                       keras.initializers.RandomUniform(seed=_seed)])

        idx = self.random_state.choice(len(self.non_conv_parameters))
        _neurons = self.non_conv_parameters[idx][0]
        _activations = self.non_conv_parameters[idx][1]
        self._computational_layer = keras.layers.Dense(units=_neurons,
                                                       activation=_activations,
                                                       kernel_initializer=kernel_initializer)








if __name__ == '__main__':
  pass
