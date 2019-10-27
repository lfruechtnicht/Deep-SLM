from tensorflow import keras
from Node import Node
from utils import *
import sys


class NeuralNetwork(object):
    """A neural network object consists of  an input Nodes, intermediate Nodes and an output Node. Together they form
    the model which is a Keras computational model."""

    def __init__(self, input_shape, n_outputs, seed=None, max_depth_cl=10, max_width_cl=4, max_depth_ff=3,
                 max_splits=3, x=None):

        self.max_depth_cl = max_depth_cl
        self.max_width_cl = max_width_cl
        self.max_depth_ff = max_depth_ff
        self.max_splits = max_splits
        self.seed_range_cl = self.max_depth_cl * self.max_width_cl
        self.seed_range_ff = self.max_depth_ff
        self.input_shape = input_shape
        self.input_node = None
        self.cl_layers = []
        self.bottleneck = None
        self.ff_layers = []
        self.output_node = None
        self.n_outputs = n_outputs
        self.model = None
        self.semantics = None
        self.random_state = check_random_state(seed)
        self._width = 1
        self.x = x

        self._eval()

        self.build_cl()
        self.build_ff()
        self._get_model()

    def _eval(self):
        """Text"""
        if not isinstance(self.input_shape, tuple):  # can also be list of nodes
            raise ValueError(
                "A neural Network mut be given an input shape as a tuple of three integers!")  # maybe go deeper
        if not isinstance(self.n_outputs, int):  # can also be list of nodes
            raise ValueError("A neural Network mut be given an output as a integers!")

    def summary(self):
        return self.model.summary()

    def build_cl(self):

        # 1 define the input node
        self.input_node = Node(is_input_node=True, computational_layer=keras.Input(shape=self.input_shape))
        # 2 add subsequent nodes
        _building_cl, _nodes_wo_connection = True, [self.input_node]
        while _building_cl:
            nodes_wo_connection = [node for node in _nodes_wo_connection if node.output_shape.ndims == 4]
            input_node = self.random_state.choice(nodes_wo_connection)
            _only_flatten = input_node.depth > self.max_depth_cl
            # case 1
            if _only_flatten:
                self.cl_layers.append(Node(input_node=input_node, computational_layer=keras.layers.Flatten()))

            _split_possible = self._width < self.max_width_cl

            # case 2
            if not _split_possible:
                self.build_next_node(input_node)

            # case 3
            if _split_possible:
                _splits_possible = min(self.max_width_cl - self._width, self.max_splits)
                splits = self.random_state.choice(_splits_possible)
                for split in range(splits + 1):
                    self._width += 1
                    self.build_next_node(input_node)

            _building_cl, _nodes_wo_connection = self._check_cl_done()

        if len(_nodes_wo_connection) > 1:
            _inputs = [node.computational_layer for node in _nodes_wo_connection]
            self.bottleneck = Node(input_node=_nodes_wo_connection,
                                   computational_layer=keras.layers.concatenate(inputs=_inputs))
        else:
            self.bottleneck = _nodes_wo_connection[0]  ##??? todo

    def build_ff(self):
        # inital as it will change over time ...

        kernel_initializer = keras.initializers.RandomNormal(seed=self.random_state.randint(sys.maxsize))
        self.ff_layers.append(Node(input_node=self.bottleneck,
                                   computational_layer=keras.layers.Dense(20,
                                                                          activation='relu',
                                                                          kernel_initializer=kernel_initializer)))

        if self.n_outputs == 1:
            activation = None
        else:
            activation = 'softmax'

        self.output_node = Node(input_node=self.ff_layers[-1],
                                computational_layer=keras.layers.Dense(self.n_outputs,
                                                                       activation=activation,
                                                                       kernel_initializer=kernel_initializer))

    def _get_model(self):
        self.model = keras.models.Model(inputs=self.input_node.computational_layer,
                                        outputs=self.output_node.computational_layer)

    def _check_cl_done(self):
        _nodes_wo_connection = [node for node in self.cl_layers if not node.out_connections]
        if not all(node.output_shape.ndims == 2 for node in _nodes_wo_connection):
            # evaluates if output tensor shape is flattened or not
            _building_cl = True  # here the nodes will be a input node after random choice
        else:
            _building_cl = False
        return _building_cl, _nodes_wo_connection  # here they will be finally concatenated and represent a Bottleneck:

    def build_next_node(self, input_node):
        # a node can be concatenated if it has the same shape
        _possible_concat_nodes = [node for node in self.cl_layers
                                  if node.output_shape.ndims > 2
                                  and input_node.output_shape[1] == node.output_shape[1]
                                  and input_node.output_shape[2] == node.output_shape[2]]

        if not self.input_node.is_input_node:
            _possible_concat_nodes.remove(input_node)
        if not _possible_concat_nodes:
            _concat_possible = False
        else:
            _concat_possible = True

        # case 1
        if not _concat_possible:
            _cl = self.random_state.choice([keras.layers.Flatten(), None])
            self.cl_layers.append(Node(input_node=input_node,
                                       computational_layer=_cl,
                                       seed=self.random_state.randint(self.seed_range_cl)))

        # case 2
        elif _concat_possible:
            _cl = self.random_state.choice([keras.layers.Flatten(), None, 'concatenate'])
            if _cl == 'concatenate':
                concat_nodes = self.random_state.choice(
                    _possible_concat_nodes, get_truncated_normal(upp=len(_possible_concat_nodes)))
                input_node = [input_node]  #
                if isinstance(concat_nodes, list):
                    input_node.extend(concat_nodes)
                    _unconnected_nodes = [node for node in concat_nodes if not node.out_connections]
                    self._width = self._width - (len(_unconnected_nodes))
                else:
                    input_node.append(concat_nodes)
                    if not concat_nodes.out_connections:
                        self._width += 1  # todo determine if this is correct ?

                _input_nodes = [node.computational_layer for node in input_node]
                self.cl_layers.append(Node(input_node=input_node,
                                           computational_layer=keras.layers.concatenate(
                                               inputs=_input_nodes),
                                           seed=self.random_state.randint(self.seed_range_cl)))
            else:
                self.cl_layers.append(Node(input_node=input_node,
                                           computational_layer=_cl,
                                           seed=self.random_state.randint(self.seed_range_cl)))
                self._width += 1  # todo is this correct ??

    def get_semantics(self):
        pass

    def get_random_mutation(self):
        _building_cl, _nodes_wo_connection = True, [self.input_node]
        #  while _building_cl:
        nodes_wo_connection = [node for node in _nodes_wo_connection if node.output_shape.ndims == 4]
        input_node = self.random_state.choice(nodes_wo_connection)

        dummy = Node(input_node=input_node,
                     computational_layer=None,
                     seed=self.random_state.randint(self.seed_range_cl))

        _next = [(Node(input_node=dummy, computational_layer=keras.layers.Flatten())), self.bottleneck]

        _nextv2 = [node.computational_layer for node in _next]
        out = Node(input_node=_next,
                   computational_layer=keras.layers.concatenate(
                       inputs=_nextv2),
                   seed=self.random_state.randint(self.seed_range_cl))

        kernel_initializer = keras.initializers.RandomNormal(seed=self.random_state.randint(sys.maxsize))
        self.ff_layers.append(Node(input_node=out,
                                   computational_layer=keras.layers.Dense(20,
                                                                          activation='relu',
                                                                          kernel_initializer=kernel_initializer)))

        if self.n_outputs == 1:
            activation = None
        else:
            activation = 'softmax'

        self.output_node_v2 = Node(input_node=self.ff_layers[-1],
                                   computational_layer=keras.layers.Dense(self.n_outputs,
                                                                          activation=activation,
                                                                          kernel_initializer=kernel_initializer))
        self.model_2 = keras.models.Model(inputs=self.input_node.computational_layer,
                                          outputs=self.output_node_v2.computational_layer)

        # _only_flatten = input_node.depth > self.max_depth_cl
        # # case 1
        # if _only_flatten:
        #     self.cl_layers.append(Node(input_node=input_node, computational_layer=keras.layers.Flatten()))
        #
        # _split_possible = self._width < self.max_width_cl
        #
        # # case 2
        # if not _split_possible:
        #     self.build_next_node(input_node)
        #
        # # case 3
        # if _split_possible:
        #     _splits_possible = min(self.max_width_cl - self._width, self.max_splits)
        #     splits = self.random_state.choice(_splits_possible)
        #     for split in range(splits + 1):
        #         self._width += 1
        #         self.build_next_node(input_node)
        #
        # _building_cl, _nodes_wo_connection = self._check_cl_done()


if __name__ == '__main__':
    Network_a = NeuralNetwork(input_shape=(64, 64, 3), n_outputs=1, seed=1)  # 3,5
    Network_a.summary()
    Network_b = NeuralNetwork(input_shape=(64, 64, 3), n_outputs=1, seed=1)  # 3,5
    Network_b.get_random_mutation()
    Network_b.model_2.summary()
