from tensorflow import keras
from tensorflow.keras import layers
from Node import Node
from utils import get_truncated_normal


class NeuralNetwork(object):
    """A neural network object consists of  an input Nodes, intermediate Nodes and an output Node. Together they form
    the model which is a Keras computational model."""

    def __init__(self, max_depth_cl, max_width_cl, max_depth_ff, max_splits, input_shape, random_state):
        self.max_depth_cl = max_depth_cl
        self.max_width_cl = max_width_cl
        self.max_depth_ff = max_depth_ff
        self.max_splits = max_splits
        self.input_shape = input_shape
        self.input_node = None
        self.cl_layers = []
        self.bottleneck = None
        self.ff_layers = []
        self.output_node = None
        self.model = None
        self.semantics = None
        self.random_state = random_state
        self.build_cl()
        self.build_ff()
        self._get_model()

    def __str__(self):
        return self.model.summary()

    def build_cl(self):


        # 1 define the input node
        self.input_node = Node(is_input_node=True, computational_layer=keras.Input(shape=self.input_shape))
        self._width = 1  # gets smaller if two unconnected nodes merge and bigger per split!
        # 2 add subsequent nodes
        _building_cl, _nodes_wo_connection = self._check_cl_done()
        while _building_cl:
            nodes_wo_connection = [node for node in _nodes_wo_connection if node.output_shape.ndims == 4]
            input_node = self.random_state.choice(nodes_wo_connection)
            _only_flatten = input_node._depth > self.max_depth_cl
            # case 1
            if _only_flatten:
                self.cl_layers.append(Node(input_node=input_node, computational_layer=keras.layers.Flatten))

            _split_possible = self._width < self.max_width_cl

            # case 2
            if not _split_possible:
                # a node can be concatenated if it has the same shape
                _possible_concat_nodes = [node for node in self.cl_layers
                                          if node.output_shape.ndims > 2
                                          and input_node.output_shape[1] == node.output_shape[1]
                                          and input_node.output_shape[2] == node.output_shape[2]]
                if not _possible_concat_nodes:
                    _concat_possible = False
                else:
                    _concat_possible = True

                # case 2.1
                if not _concat_possible:
                    _cl = self.random_state.choice([keras.layers.Flatten, None])
                    self.cl_layers.append(Node(input_node=input_node, computational_layer=_cl))

                # case 2.2
                elif _concat_possible:
                    _cl = self.random_state.choice([keras.layers.Flatten, None, keras.layers.concatenate])
                    if _cl == keras.layers.concatenate:
                        concat_nodes = self.random_state.choice(
                            _possible_concat_nodes, get_truncated_normal(upp=len(_possible_concat_nodes)))
                        input_node = [input_node]
                        input_node.extend(concat_nodes)

                        # set self._width =-1 if for each two unconnected nodes that merged
                        _unconnected_nodes = [node for node in concat_nodes if not node.out_connections]
                        self._width = self._width - (len(_unconnected_nodes))

                    self.cl_layers.append(Node(input_node=input_node, computational_layer=_cl))

            # case 3
            if _split_possible:
                _splits_possible = min(self.max_width_cl - self._width, self.max_splits)
                splits = self.random_state.choice(_splits_possible)
                for split in range(splits):
                    self._width += 1
                    # now per split we do the same things as we would do for only one split
                    _possible_concat_nodes = [node for node in self.cl_layers
                                              if node.output_shape.ndims > 2
                                              and input_node.output_shape[1] == node.output_shape[1]
                                              and input_node.output_shape[2] == node.output_shape[2]]
                    if not _possible_concat_nodes:
                        _concat_possible = False
                    else:
                        _concat_possible = True

                    # case 3.1
                    if not _concat_possible:
                        _cl = self.random_state.choice([keras.layers.Flatten, None])
                        self.cl_layers.append(Node(input_node=input_node, computational_layer=_cl))

                    # case 3.2
                    elif _concat_possible:
                        _cl = self.random_state.choice([keras.layers.Flatten, None, keras.layers.concatenate])
                        if _cl == keras.layers.concatenate:
                            concat_nodes = self.random_state.choice(
                                _possible_concat_nodes, get_truncated_normal(upp=len(_possible_concat_nodes)))
                            input_node = [input_node]
                            input_node.extend(concat_nodes)

                            # set self._width =-1 if for each two unconnected nodes that merged
                            _unconnected_nodes = [node for node in concat_nodes if not node.out_connections]
                            self._width = -1 * len(_unconnected_nodes)

                        self.cl_layers.append(Node(input_node=input_node, computational_layer=_cl))

            _building_cl, _nodes_wo_connection = self._check_cl_done()
        self.bottleneck = Node(_nodes_wo_connection, layers.concatenate)

    def build_ff(self):
        #inital as it will change over time ...
        self.ff_layers.append(Node(input_node=self.bottleneck,
                                   computational_layer=keras.layers.Dense(20, activation='relu')))
        self.output_node = Node(input_node=self.ff_layers[-1],
                                computational_layer=keras.layers.Dense(20, activation='sigmoid'))

    def _get_model(self):
        self.model = keras.models.Model(inputs=self.input_node.computational_layer,
                                        outputs=self.output_node.computational_layer)

    def _check_cl_done(self):
        _nodes_wo_connection = [node for node in self.cl_layers if node.out_connections is None]
        if not all(node.output_shape.ndims == 2 for node in _nodes_wo_connection):
            # evaluates if output tensor shape is flattened or not
            _building_cl = True  # here the nodes will be a input node after random choice
        else:
            _building_cl = False
        return _building_cl, _nodes_wo_connection  # here they will be finally concatenated and represent a Bottleneck
