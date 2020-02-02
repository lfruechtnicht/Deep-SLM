import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from DeepSemanticLearningMachine.Node import Node
from utils import *
import sys
from itertools import product


class NeuralNetwork(object):
    """A neural network object consists of  an input Nodes, layers and an output Node. Together they form
    the model which is a Keras computational model. The network can be mutated such that additional layers are added.
     A secondary graph layer is introduced to work with intermediary outputs to not recalculate the whole network.

     x_train np.array Training images
     y_train np.array Training lables
     input_shape tuple dimensions of the images
     n_outputs number of classes
     metric any instance of class metric
     seed seed or Random State instance to be reproducible
     layer_parameters: parameters for all layers as recieved from the DeepSLM
     conv_parameters: list containing all possible combinations of parameters for keras conv layers
     pool_parameters: list containing all possible combinations of parameters for keras pool layers
     non_conv_parameters: list containing all possible combinations of parameters for keras dense layers
     max_depth_cl int max number of cl excluding concatenations
     max_width_cl int max number of parralel layers in one mutation
     max_depth_ff int
     max_width_ff int
     max_splits int max number of differnt layers following one node (restricted by max_depths)
     mutation_level int number of mutations this network has seen
     output_node keras.layer output layer of the model
     model keras.model.Model the computational graph which is mutated
     semantics np.array the output predictions of the model
     semantics_size int the size of the semantics accupied in memory #
     _width int current with of the network #
     initial_input_node Node which is exposed to the raw dataset
     layers Dict containing per mutation level a list of nodes
    """

    def __init__(self,
                 x_train,
                 y_train,
                 input_shape,
                 n_outputs,
                 metric,
                 seed,
                 layer_parameters,
                 conv_parameters,
                 pool_parameters,
                 non_conv_parameters,
                 max_depth_cl=20,
                 max_width_cl=1,
                 max_depth_non_conv=4,
                 max_width_non_conv=3,
                 max_splits=3,
                 mutation_level=0,
                 pre_output_node=None,
                 output_node=None,
                 model=None,
                 semantics=None,
                 semantics_size=0,
                 fitness=None,
                 initial_input_node=None,
                 layers=None,
                 validation_data=None
                 ):

        self.random_state = check_random_state(seed)
        self.input_shape = input_shape
        self.n_outputs = n_outputs
        self.metric = metric
        self.layer_parameters = layer_parameters

        self.conv_parameters: list = conv_parameters
        self.pool_parameters = pool_parameters
        self.non_conv_parameters = non_conv_parameters

        self.max_depth_cl = max_depth_cl
        self.max_width_cl = max_width_cl
        self.max_depth_non_conv = max_depth_non_conv
        self.max_width_non_conv = max_width_non_conv
        self.max_splits = max_splits
        self.mutation_level = mutation_level
        self.x_train = x_train
        self.y_train = y_train
        self.validation_data = validation_data
        self.pre_output_node = pre_output_node
        self.output_node = output_node
        self.model = model
        self.semantics = semantics
        self.semantics_size = semantics_size
        self.fitness = fitness
        self.current_with = 1

        self.initial_input_node = initial_input_node
        self.layers = layers
        if self.initial_input_node is None:
            self.initial_input_node = Node(mutation_level=self.mutation_level,
                                           is_input_node=True,
                                           seed=self.random_state,
                                           _computational_layer=keras.Input(shape=self.input_shape),
                                           input_shape=self.input_shape,
                                           )
            self.layers = {"conv": {self.mutation_level: []},
                           "non_conv": {self.mutation_level: []}}
            self.pre_output_node = {self.mutation_level: Node}
            self.initialize_network()

    def __repr__(self):
        return str(f"NeuralNetwork, {self.metric.name}: {self.fitness:.4f}, Mutations: {self.mutation_level}, "
                   f"MemoUsage: {self.semantics_size:.2f}mb")

    def __copy__(self):
        return self._copy()

    def copy(self):
        return self._copy()

    def _copy(self):
        """returns a copy of the Network"""

        seed = self.random_state
        input_shape = self.input_shape
        n_outputs = self.n_outputs
        metric = self.metric
        layer_parameters = self.layer_parameters
        conv_parameters = self.conv_parameters
        pool_parameters = self.pool_parameters
        non_conv_parameters = self.non_conv_parameters
        max_depth_cl = self.max_depth_cl
        max_width_cl = self.max_width_cl
        max_depth_non_conv = self.max_depth_non_conv
        max_width_non_conv = self.max_width_non_conv
        max_splits = self.max_splits
        mutation_level = self.mutation_level
        x_train = self.x_train
        y_train = self.y_train
        validation_data = self.validation_data
        pre_output_node = self.pre_output_node
        output_node = self.output_node
        semantics = self.semantics
        semantics_size = self.semantics_size
        initial_input_node = self.initial_input_node
        layers = self.layers.copy()
        fitness = self.fitness
        return NeuralNetwork(seed=seed,
                             input_shape=input_shape,
                             n_outputs=n_outputs,
                             metric=metric,
                             layer_parameters=layer_parameters,
                             conv_parameters=conv_parameters,
                             pool_parameters=pool_parameters,
                             non_conv_parameters=non_conv_parameters,
                             max_depth_cl=max_depth_cl,
                             max_width_cl=max_width_cl,
                             max_depth_non_conv=max_depth_non_conv,
                             max_width_non_conv=max_width_non_conv,
                             max_splits=max_splits,
                             mutation_level=mutation_level,
                             x_train=x_train,
                             y_train=y_train,
                             pre_output_node=pre_output_node,
                             output_node=output_node,
                             semantics=semantics,
                             semantics_size=semantics_size,
                             fitness=fitness,
                             initial_input_node=initial_input_node,
                             layers=layers,
                             validation_data=validation_data)

    def summary(self):
        return self.model.summary()

    def initialize_network(self):
        """Builds the first network and sets semantics and determines fitness"""
        self.build_random_layers(self.initial_input_node)
        self.get_semantics_initial_nodes()
        self.semantics = self.output_node.semantics
        self.fitness = self._evaluate()

    def build_random_layers(self, starting_node, concatenation_restricted=False):
        self.build_random_conv(starting_node, concatenation_restricted)
        self.build_fixed_dense()

    # def _initialize_network(self):
    #     """Builds the first network and sets semantics and determines fitness"""
    #     self._build_random_layers(self.initial_input_node)
    #     self.get_semantics_initial_nodes()
    #     self.semantics = self.output_node.semantics
    #     self.fitness = self._evaluate()
    #
    # def _build_random_layers(self, starting_node, concatenation_restricted=False):
    #     if len(starting_node.output_shape) == 4:
    #         self.build_random_conv(starting_node, concatenation_restricted)
    #         self.build_random_non_conv(self.layers["conv"][self.mutation_level][-1], concatenation_restricted=True)
    #         self.build_final_nodes()
    #     else:
    #         self.build_random_non_conv(starting_node, concatenation_restricted)
    #         self.build_final_nodes()

    def build_random_conv(self, starting_node, concatenation_restricted=False):
        """

        :param starting_node: of type Node
        :param concatenation_restricted:
        :return:
        """

        # initialize two variables
        building_conv, nodes_wo_connection, layer_type = True, [starting_node], "conv"

        while building_conv:  # True until only flatten() is chosen as a layer or reaches max_depth
            nodes_wo_connection = [node for node in nodes_wo_connection if len(node.output_shape) == 4]  # Conv nodes!
            to_connect_with_node = self.random_state.choice(nodes_wo_connection)  # one input to the next node

            only_flatten = to_connect_with_node.depth >= self.max_depth_cl or to_connect_with_node.only_flatten_test()
            # case 1
            if only_flatten:
                self.layers[layer_type][self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                                         conv_parameters=self.conv_parameters,
                                                                         pool_parameters=self.pool_parameters,
                                                                         seed=self.random_state,
                                                                         input_node=to_connect_with_node,
                                                                         _computational_layer=keras.layers.Flatten()))
            else:
                no_splitting = self.current_with >= self.max_width_cl

                # case 2
                if no_splitting:
                    self.build_next_conv_node(to_connect_with_node, concatenation_restricted)
                # case 3
                else:
                    n_possible_splits: int = min(self.max_width_cl - self.current_with, self.max_splits)
                    splits = self.random_state.choice(list(range(n_possible_splits)))
                    for split in range(splits + 1):
                        self.current_with += 1
                        self.build_next_conv_node(to_connect_with_node, concatenation_restricted)

            building_conv, nodes_wo_connection = self.check_conv_done()

        if len(nodes_wo_connection) > 1:  # concatenate all parallel layers to have one input for the dense layers
            self.layers[layer_type][self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                                     conv_parameters=self.conv_parameters,
                                                                     pool_parameters=self.pool_parameters,
                                                                     seed=self.random_state,
                                                                     input_node=nodes_wo_connection,
                                                                     _computational_layer=keras.layers.Concatenate()))
        self.current_with = 1

    def check_conv_done(self):
        nodes_wo_connection = [node for node in self.layers["conv"][self.mutation_level] if not node.out_connections]
        if not all(len(node.output_shape) == 2 for node in nodes_wo_connection):  # if not all nodes are flat
            # evaluates if output tensor shape is flattened or not
            building_cl = True  # here the nodes will be a input node after random choice
        else:
            building_cl = False  # here they will be finally concatenated and represent a Bottleneck:
        return building_cl, nodes_wo_connection

    def build_next_conv_node(self, to_connect_with_node, concatenation_restricted):

        if concatenation_restricted:  # just build a node
            _computational_layer = self.random_state.choice([keras.layers.Flatten(), None])
            self.layers["conv"][self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                                 conv_parameters=self.conv_parameters,
                                                                 pool_parameters=self.pool_parameters,
                                                                 input_node=to_connect_with_node,
                                                                 _computational_layer=_computational_layer,
                                                                 seed=self.random_state,
                                                                 layer_type="conv"))
        else:
            # a node can be concatenated if it has the same shape
            all_nodes = self.get_all_nodes_layer_type("conv")  # all possible nodes of the network of the layer type
            possible_concat_nodes = [node for node in all_nodes  # need to match the outer node in shape
                                     if len(node.output_shape) > 2  # todo make more efficient on node level
                                     and to_connect_with_node.output_shape[1] == node.output_shape[1]
                                     and to_connect_with_node.output_shape[2] == node.output_shape[2]]

            if self.layers["conv"][self.mutation_level]:
                possible_concat_nodes.remove(to_connect_with_node)
            if not possible_concat_nodes:  # if there are no nodes with the same shape as the input node
                concat_possible = False
            else:
                concat_possible = True

            # case 1
            if not concat_possible:
                _computational_layer = self.random_state.choice([keras.layers.Flatten(), None])
                self.layers["conv"][self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                                     conv_parameters=self.conv_parameters,
                                                                     pool_parameters=self.pool_parameters,
                                                                     input_node=to_connect_with_node,
                                                                     _computational_layer=_computational_layer,
                                                                     seed=self.random_state))

            # case 2
            elif concat_possible:
                _computational_layer = self.random_state.choice([keras.layers.Flatten(),
                                                                 None,
                                                                 keras.layers.Concatenate()])
                if isinstance(_computational_layer, keras.layers.Concatenate):
                    concat_nodes = self.random_state.choice(possible_concat_nodes,
                                                            # basically chose x where x is distributed as half a gaussian but it needs revise
                                                            get_truncated_normal(upp=len(
                                                                possible_concat_nodes)))  # todo change this as it is not just want you want please see with upp 3 it is very likly

                    to_connect_with_node = [to_connect_with_node]

                    if isinstance(concat_nodes, list):
                        to_connect_with_node.extend(concat_nodes)
                        _unconnected_nodes = [node for node in concat_nodes if not node.out_connections]
                        self.current_with = self.current_with - (
                            len(_unconnected_nodes))  # if i concatenate with unconnected nodes
                    else:
                        to_connect_with_node.append(concat_nodes)
                        if not concat_nodes.out_connections:
                            self.current_with -= 1

                    self.layers["conv"][self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                                         conv_parameters=self.conv_parameters,
                                                                         pool_parameters=self.pool_parameters,
                                                                         input_node=to_connect_with_node,
                                                                         _computational_layer=_computational_layer,
                                                                         seed=self.random_state))
                else:
                    self.layers["conv"][self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                                         conv_parameters=self.conv_parameters,
                                                                         pool_parameters=self.pool_parameters,
                                                                         input_node=to_connect_with_node,
                                                                         _computational_layer=_computational_layer,
                                                                         seed=self.random_state))
                    self.current_with += 1

    def build_fixed_dense(self):

        """Builds the dense layer part
                for the moment it is static but could easily be adapted to work as a SLM"""
        kernel_initializer = keras.initializers.RandomNormal(seed=self.random_state.randint(sys.maxsize))
        self.layers["non_conv"][self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                                 seed=self.random_state,
                                                                 input_node=self.layers["conv"][self.mutation_level][
                                                                     -1],
                                                                 _computational_layer=keras.layers.Dense(
                                                                     20,
                                                                     activation='relu',
                                                                     kernel_initializer=kernel_initializer)))

        self.build_final_nodes()

    def build_randome_dense(self):

        """Builds the dense layer part
                for the moment it is static but could easily be adapted to work as a SLM"""
        self.layers["non_conv"][self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                                 seed=self.random_state,
                                                                 input_node=self.layers["conv"][self.mutation_level][
                                                                     -1],
                                                                 layer_type="non_conv"
                                                                 ))

        self.build_final_nodes()

    def get_last_nodes(self):
        return [sublist[-1] for sublist in self.layers["non_conv"].values()]

    def build_random_non_conv(self, starting_node, concatenation_restricted=False):

        # skip connections?
        # how many layer
        building_non_conv, nodes_wo_connection, layer_type = True, [starting_node], "non_conv"
        while building_non_conv:
            to_connect_with_node = self.random_state.choice(nodes_wo_connection)  # one input to the next node

            # case 2
            if self.current_with >= self.max_width_non_conv:
                self.build_next_non_conv_node(to_connect_with_node, concatenation_restricted)
            # case 3
            else:
                n_possible_splits: int = min(self.max_width_non_conv - self.current_with, self.max_splits)
                splits = self.random_state.choice(list(range(n_possible_splits)))
                for split in range(splits + 1):
                    self.current_with += 1
                    self.build_next_non_conv_node(to_connect_with_node, concatenation_restricted)

            building_non_conv, nodes_wo_connection = self.check_non_conv_done()

        if len(nodes_wo_connection) > 1:  # concatenate all parallel layers to have one input for the dense layers
            self.layers[layer_type][self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                                     non_conv_parameters=self.non_conv_parameters,
                                                                     seed=self.random_state,
                                                                     input_node=nodes_wo_connection,
                                                                     _computational_layer=keras.layers.Concatenate(),
                                                                     layer_type="non_conv"))

        # case dependant output  # todo separate classes for separate tasks

    def check_non_conv_done(self):
        nodes_wo_conn = [node for node in self.layers["non_conv"][self.mutation_level] if not node.out_connections]

        return self.layers["non_conv"][self.mutation_level][-1].depth <= self.max_depth_non_conv, nodes_wo_conn

    def build_next_non_conv_node(self, to_connect_with_node, concatenation_restricted=True):

        if concatenation_restricted:  # just build a node
            self.layers["non_conv"][self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                                     non_conv_parameters=self.non_conv_parameters,
                                                                     input_node=to_connect_with_node,
                                                                     _computational_layer=None,
                                                                     seed=self.random_state,
                                                                     layer_type="non_conv"))
        else:
            # a node can be concatenated if it has the same shape
            possible_concat_nodes = [item for sublist in self.layers["non_conv"].values() for item in sublist]
            possible_concat_nodes.append(self.initial_input_node)
            # all possible nodes of the network of the layer type
            if self.layers["non_conv"][self.mutation_level]:  # todo if flatten layer
                possible_concat_nodes.remove(to_connect_with_node)

            if not possible_concat_nodes:  # if there are no nodes with the same shape as the input node
                concat_possible = False
            else:
                concat_possible = True

            # case 1
            if not concat_possible:
                self.layers["non_conv"][self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                                         non_conv_parameters=self.non_conv_parameters,
                                                                         input_node=to_connect_with_node,
                                                                         _computational_layer=None,
                                                                         seed=self.random_state,
                                                                         layer_type="non_conv"))

            # case 2
            elif concat_possible:
                _computational_layer = self.random_state.choice([None,
                                                                 keras.layers.Concatenate()])
                if isinstance(_computational_layer, keras.layers.Concatenate):
                    concat_nodes = self.random_state.choice(possible_concat_nodes,
                                                            # basically chose x where x is distributed as half a gaussian but it needs revise
                                                            get_truncated_normal(upp=len(
                                                                possible_concat_nodes)))  # todo change this as it is not just want you want please see with upp 3 it is very likly

                    to_connect_with_node = [to_connect_with_node]

                    if isinstance(concat_nodes, list):
                        to_connect_with_node.extend(concat_nodes)
                        _unconnected_nodes = [node for node in concat_nodes if not node.out_connections]
                        self.current_with = self.current_with - (
                            len(_unconnected_nodes))  # if i concatenate with unconnected nodes
                    else:
                        to_connect_with_node.append(concat_nodes)
                        if not concat_nodes.out_connections:
                            self.current_with -= 1

                    self.layers["non_conv"][self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                                             non_conv_parameters=self.non_conv_parameters,
                                                                             input_node=to_connect_with_node,
                                                                             _computational_layer=_computational_layer,
                                                                             seed=self.random_state,
                                                                             layer_type="non_conv"))
                else:
                    self.layers["non_conv"][self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                                             non_conv_parameters=self.non_conv_parameters,
                                                                             input_node=to_connect_with_node,
                                                                             _computational_layer=_computational_layer,
                                                                             seed=self.random_state,
                                                                             layer_type="non_conv"))
                    self.current_with += 1

    def build_final_nodes(self):  # Here Classification
        """
        Because the Crossentropy which is the usual loss for classification requires each prediction to resemble a
        propability distribution we need to ensure that each output prediction is within [0,1] with sum = 1
        :return:
        """

        if self.metric.type is "regression":
            activations = ["linear", "linear"]
        elif self.metric.type is "classification":
            activations = ["relu", "softmax"]
        else:
            raise ValueError

        kernel_initializer = keras.initializers.RandomNormal(seed=self.random_state.randint(sys.maxsize))
        self.layers["non_conv"][self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                                 conv_parameters=self.conv_parameters,
                                                                 pool_parameters=self.pool_parameters,
                                                                 input_node=
                                                                 self.layers["non_conv"][self.mutation_level][-1],
                                                                 seed=self.random_state,
                                                                 _computational_layer=keras.layers.Dense(
                                                                     self.n_outputs,
                                                                     activation=activations[0],
                                                                     kernel_initializer=kernel_initializer)))
        if self.mutation_level == 0:
            self.pre_output_node[self.mutation_level] = self.layers["non_conv"][self.mutation_level][
                -1]

            # output for the initial network is the last node
        else:
            last_pre_output = self.pre_output_node[self.mutation_level - 1]
            last_node_current = self.layers["non_conv"][self.mutation_level][-1]
            self.pre_output_node[self.mutation_level] = (Node(mutation_level=self.mutation_level,
                                                              conv_parameters=self.conv_parameters,
                                                              pool_parameters=self.pool_parameters,
                                                              seed=self.random_state,
                                                              input_node=[last_pre_output, last_node_current],
                                                              _computational_layer=keras.layers.Add()))
            # maybe also using average or something else?
        self.output_node = Node(mutation_level=self.mutation_level,
                                conv_parameters=self.conv_parameters,
                                pool_parameters=self.pool_parameters,
                                input_node=self.pre_output_node[self.mutation_level],
                                seed=self.random_state,

                                _computational_layer=keras.layers.Dense(
                                    self.n_outputs,
                                    activation=activations[1],
                                    kernel_initializer=keras.initializers.Ones(),
                                    bias_initializer=keras.initializers.Zeros()))
        self.output_node = self.pre_output_node[self.mutation_level]

    def get_semantics_initial_nodes(self):
        """builds a model to return the semantics for all nodes and saves them to each node"""
        input_node = self.initial_input_node.computational_layer
        output_nodes = [node for node in self.layers["conv"][
            self.mutation_level]]
        output_nodes.extend([node for node in self.layers["non_conv"][
            self.mutation_level]])
        output_nodes.append(self.pre_output_node[self.mutation_level])
        output_nodes.append(self.output_node)
        output_nodes_layer = [node.computational_layer for node in output_nodes]

        model_for_semantics = keras.models.Model(inputs=input_node, outputs=output_nodes_layer)
        semantics = model_for_semantics.predict(self.x_train)
        self.semantics_size += sum([sem.nbytes for sem in semantics]) / 1.E6
        [setattr(output_nodes[idx], 'semantics', i) for idx, i in enumerate(semantics)]
        self.initial_input_node.semantics = self.x_train

    def isolated_mutation(self):
        """Mutate the network such that a graph with only one touch-point is added to the network
        returns the network, contains more than one neuron per layer """
        self.mutation_level += 1
        self.layers["conv"][self.mutation_level] = []
        self.layers["non_conv"][self.mutation_level] = []
        mutation_node = self.random_mutation_node()  # choose a staring node
        self.build_random_layers(starting_node=mutation_node, concatenation_restricted=True)
        self.get_semantics_mutation_nodes()
        self.semantics = self.output_node.semantics
        self.fitness = self._evaluate()
        return self

    def isolated_one_filter_mutation(self):
        """Mutate the network such that a graph with only one touch-point is added to the network
        returns the network, adds only one neuron per layer """
        self.conv_parameters, self.non_conv_parameters = self._set_parameters_one_filter()
        self.isolated_mutation()
        return self

    def connected_mutation(self):
        """Mutate the network such that a graph with only one touch-point is added to the network
        returns the network, contains more than one neuron per layer """
        self.mutation_level += 1
        self.layers["conv"][self.mutation_level] = []
        self.layers["non_conv"][self.mutation_level] = []
        mutation_node = self.random_mutation_node()  # choose a staring node
        self.build_random_layers(starting_node=mutation_node, concatenation_restricted=False)
        self.get_semantics_mutation_nodes()
        self.semantics = self.output_node.semantics
        self.fitness = self._evaluate()
        return self

    def connected_one_filter_mutation(self):
        """Mutate the network such that a graph with only one touch-point is added to the network
        returns the network, adds only one neuron per layer """
        self.conv_parameters, self.non_conv_parameters = self._set_parameters_one_filter()
        self.connected_mutation()
        return self

    def get_semantic_inputs(self):
        """searches if any node in the current mutation level has inputs from previous semantics and returns a tuple of
        with a list of the semantic inputs resembling a keras.layer.input and the associated data to the input"""
        semantic_input_nodes = flatten([node.semantic_input_node for node in self.layers["conv"][self.mutation_level]
                                        if node.semantic_input])
        semantic_input_nodes.extend(flatten([node.semantic_input_node for node in
                                             self.layers["non_conv"][self.mutation_level] if node.semantic_input]))
        semantic_input_nodes.append(self.pre_output_node[self.mutation_level].semantic_input_node[0])
        semantic_data = flatten([node.input_data for node in self.layers["conv"][self.mutation_level]
                                 if node.semantic_input])
        semantic_data.extend(flatten([node.input_data for node in self.layers["non_conv"][self.mutation_level]
                                      if node.semantic_input]))
        semantic_data.append(self.pre_output_node[self.mutation_level].input_data[0])

        return semantic_input_nodes, semantic_data

    def get_semantics_mutation_nodes(self):
        """gets the semantics for all newly added nodes in this mutation"""
        semantic_input_nodes, semantic_data = self.get_semantic_inputs()  # get the input nodes and the data
        output_nodes = [node for node in self.layers["conv"][self.mutation_level]]
        output_nodes.extend([node for node in self.layers["non_conv"][self.mutation_level]])
        output_nodes.append(self.pre_output_node[self.mutation_level])
        output_nodes.append(self.output_node)
        output_nodes_semantics = [node.semantics_computational_layer for node in output_nodes]
        model_for_semantics = keras.models.Model(inputs=semantic_input_nodes,
                                                 outputs=output_nodes_semantics)  # build the model_for_semantics
        semantics = model_for_semantics.predict(semantic_data)
        self.semantics_size += sum([sem.nbytes for sem in semantics]) / 1.E6
        [setattr(output_nodes[idx], 'semantics', i) for idx, i in enumerate(semantics)]

    def random_mutation_node(self):
        """:returns any Node"""
        all_nodes = self.get_all_nodes()
        mutation_node = self.random_state.choice([node for node in all_nodes if len(node.output_shape) > 2])
        return mutation_node

    def get_all_nodes(self):
        """ :returns list of all Nodes"""

        all_nodes = [item for sublist in self.layers["conv"].values() for item in sublist]
        all_nodes.extend([item for sublist in self.layers["non_conv"].values() for item in sublist])
        all_nodes.append(self.initial_input_node)
        return all_nodes

    def get_all_nodes_layer_type(self, layer_type=None):
        """ :returns list of all Nodes depending on the layer_type"""
        all_nodes = [item for sublist in self.layers[layer_type].values() for item in sublist]
        all_nodes.append(self.initial_input_node)
        return all_nodes

    def _get_model(self):
        """ builds the mode"""
        self.model = keras.models.Model(inputs=self.initial_input_node.computational_layer,
                                        outputs=self.output_node.computational_layer)

    def _evaluate(self):
        """"evaluates the network while training"""
        return self.metric.evaluate(prediction=self.semantics, target=self.y_train)

    def set_model(self):
        """builds the keras model only needed for validation"""
        self.model = Model(inputs=self.initial_input_node.computational_layer,
                           outputs=self.output_node.computational_layer)

    def evaluate(self):
        if self.validation_data is None:
            raise ValueError(
                f"Need to pass Validation data, in the form of [xtest, ytest]. Cant validate on {self.validation_data}")
        self.set_model()
        y_pred = self.model.predict(self.validation_data[0])
        del self.model  # not needed anymore
        """"evaluates the network on test data"""
        return self.metric.evaluate(prediction=y_pred, target=self.validation_data[1])

    def predict(self, data):
        return self.model.predict(data)

    def _set_parameters_one_filter(self):
        filters, kernel_sizes, strides, padding, activations, pool_size, neurons = self.layer_parameters.values()
        _conv_parameters = [[1], kernel_sizes, strides, padding, activations]
        _non_conv_parameters = [[1], activations]
        return list(product(*_conv_parameters)), list(product(*_non_conv_parameters))


if __name__ == '__main__':
    Network_a = NeuralNetwork(input_shape=(5,), n_outputs=1, seed=1)  # 3,5
    Network_a.summary()
    Network_b = NeuralNetwork(input_shape=(64, 64, 3), n_outputs=1, seed=1)  # 3,5
    Network_b  # .get_random_mutation()
    # Network_b.model_2.summary()
