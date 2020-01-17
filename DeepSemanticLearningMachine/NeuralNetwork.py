from tensorflow import keras
from tensorflow.keras.models import Model
from DeepSemanticLearningMachine.Node import Node
from utils import *
import sys
from copy import copy


class NeuralNetwork(object):  # todo inheret from individual
    """A neural network object consists of  an input Nodes, layers and an output Node. Together they form
    the model which is a Keras computational model. The network can be mutated such that additional layers are added.
     A secondary graph layer is introduced to work with intermediary outputs to not recalculate the whole network.

     x_train np.array Training images
     y_train np.array Training lables
     input_shape tuple dimensions of the images
     n_outputs number of classes
     seed seed or Random State instance to be reproducible
     x_test np.array not specified ... I think i dont need it #todo
     max_depth_cl int max number of cl excluding concatenations
     max_width_cl int max number of parralel layers in one mutation
     max_depth_ff int
     max_width_ff int
     max_splits int max number of differnt layers following one node (restricted by max_depths)
     mutation_level int number of mutations this network has seen
     output_node keras.layer output layer of the model
     model keras.model.Model the computational graph which is mutated
     semantics np.array the output predictions of the model
     semantics_size int the size of the semantics accupied in memory #todo make sure this is ok
     _width int current with of the network # todo check if this is ok for the mutations too
     initial_input_node Node which is exposed to the raw dataset
     layers Dict containing per mutation level a list of nodes
    """

    def __init__(self,
                 x_train,
                 y_train,
                 input_shape,
                 n_outputs,
                 seed=None,
                 x_test=None,
                 max_depth_cl=20,
                 max_width_cl=1,
                 max_depth_ff=3,
                 max_width_ff=1,
                 max_splits=3,
                 mutation_level=0,
                 output_node=None,
                 model=None,
                 semantics=None,
                 semantics_size=0,
                 _width=1,
                 initial_input_node=None,
                 layers=None):# todo add real metric

        self.random_state = check_random_state(seed)
        self.input_shape = input_shape
        self.n_outputs = n_outputs

        self.max_depth_cl = max_depth_cl
        self.max_width_cl = max_width_cl
        self.max_depth_ff = max_depth_ff
        self.max_width_ff = max_width_ff
        self.max_splits = max_splits
        self.mutation_level = mutation_level
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.output_node = output_node
        self.model = model
        self.semantics = semantics
        self.semantics_size = semantics_size
        self._width = _width

        self.fitness = None
        self._eval()

        self.initial_input_node = initial_input_node
        self.layers = layers
        if self.initial_input_node is None:
            self.initial_input_node = Node(mutation_level=self.mutation_level,
                                           is_input_node=True,
                                           _computational_layer=keras.Input(shape=self.input_shape),
                                           input_shape=self.input_shape)
            self.layers = {self.mutation_level: []}
            self.initialize_network()

    def __repr__(self):
        return str(f"NeuralNetwork, Fitness: {self.fitness}, Mutations: {self.mutation_level},  "
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
        max_depth_cl = self.max_depth_cl
        max_width_cl = self.max_width_cl
        max_depth_ff = self.max_depth_ff
        max_width_ff = self.max_width_ff
        max_splits = self.max_splits
        mutation_level = self.mutation_level
        x_train = self.x_train
        y_train = self.y_train
        x_test = self.x_test
        output_node = self.output_node
        semantics = self.semantics
        semantics_size = self.semantics_size
        initial_input_node = self.initial_input_node
        layers = self.layers.copy()
        return NeuralNetwork(seed=seed,
                             input_shape=input_shape,
                             n_outputs=n_outputs,
                             max_depth_cl=max_depth_cl,
                             max_width_cl=max_width_cl,
                             max_depth_ff=max_depth_ff,
                             max_width_ff=max_width_ff,
                             max_splits=max_splits,
                             mutation_level=mutation_level,
                             x_train=x_train,
                             y_train=y_train,
                             x_test=x_test,
                             output_node=output_node,
                             semantics=semantics,
                             semantics_size=semantics_size,
                             initial_input_node=initial_input_node,
                             layers=layers)

    def _eval(self):
        """actually not needed because end user will not be exposed to this"""
        if not isinstance(self.input_shape, tuple):  # can also be list of nodes
            raise ValueError(
                "A neural Network mut be given an input shape as a tuple of three integers!")  # maybe go deeper
        if not isinstance(self.n_outputs, int):  # can also be list of nodes
            raise ValueError("A neural Network mut be given an output as a integers!")

    def summary(self):
        return self.model.summary()

    def initialize_network(self):
        """Builds the first network and sets semantics and determines fitness"""
        self.build_random_layers(self.initial_input_node)
        self.get_semantics_initial_nodes()
        self.semantics = self.output_node.semantics
        self.fitness = self._evaluate()
        self.mutation_level += 1

    def build_random_layers(self, starting_node,
                            concatenation_restricted=False):  # concatenation of nodes with the same shape
        # initialize two variables
        building_cl, nodes_wo_connection = True, [starting_node]
        while building_cl:  # True until only flatten() is chosen as a layer or reaches max_depth
            nodes_wo_connection = [node for node in nodes_wo_connection if len(node.output_shape) == 4]  # Conv nodes!
            to_connect_with_node = self.random_state.choice(nodes_wo_connection)  # one input to the next node

            only_flatten = to_connect_with_node.depth >= self.max_depth_cl or to_connect_with_node.only_flatten_test()
            # case 1
            if only_flatten:
                self.layers[self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                             input_node=to_connect_with_node,
                                                             _computational_layer=keras.layers.Flatten()))
            else:
                no_splitting = self._width >= self.max_width_cl

                # case 2
                if no_splitting:
                    self.build_next_node(to_connect_with_node, concatenation_restricted)
                # case 3
                else:
                    n_possible_splits: int = min(self.max_width_cl - self._width, self.max_splits)
                    splits = self.random_state.choice(list(range(n_possible_splits)))
                    for split in range(splits + 1):
                        self._width += 1
                        self.build_next_node(to_connect_with_node, concatenation_restricted)

            building_cl, nodes_wo_connection = self.check_cl_done()

        if len(nodes_wo_connection) > 1:  # concatenate all parallel layers to have one input for the dense layers
            self.layers[self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                         input_node=nodes_wo_connection,
                                                         _computational_layer=keras.layers.Concatenate()))

        self.build_dense()

    def check_cl_done(self):
        nodes_wo_connection = [node for node in self.layers[self.mutation_level] if not node.out_connections]
        if not all(len(node.output_shape) == 2 for node in nodes_wo_connection):  # if not all nodes are flat
            # evaluates if output tensor shape is flattened or not
            building_cl = True  # here the nodes will be a input node after random choice
        else:
            building_cl = False  # here they will be finally concatenated and represent a Bottleneck:
        return building_cl, nodes_wo_connection

    def build_next_node(self, to_connect_with_node, concatenation_restricted):

        if concatenation_restricted:  # just build a node
            _computational_layer = self.random_state.choice([keras.layers.Flatten(), None])
            self.layers[self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                         input_node=to_connect_with_node,
                                                         _computational_layer=_computational_layer,
                                                         seed=self.random_state))
        else:
            # a node can be concatenated if it has the same shape
            all_cl_nodes = self.get_all_nodes()  # all possible nodes of the network
            possible_concat_nodes = [node for node in all_cl_nodes  # need to match the outer node in shape
                                     if len(node.output_shape) > 2
                                     and to_connect_with_node.output_shape[1] == node.output_shape[1]
                                     and to_connect_with_node.output_shape[2] == node.output_shape[2]]

            if not to_connect_with_node.is_input_node: # todo revise this
                possible_concat_nodes.remove(to_connect_with_node)
            if not possible_concat_nodes:  # are there other nodes with the same shape ?
                concat_possible = False
            else:
                concat_possible = True

            # case 1
            if not concat_possible:
                _computational_layer = self.random_state.choice([keras.layers.Flatten(), None])
                self.layers[self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                             input_node=to_connect_with_node,
                                                             _computational_layer=_computational_layer,
                                                             seed=self.random_state))

            # case 2
            elif concat_possible:
                _computational_layer = self.random_state.choice([keras.layers.Flatten(),
                                                                 None,
                                                                 keras.layers.Concatenate()])
                if isinstance(_computational_layer, keras.layers.Concatenate):
                    concat_nodes = self.random_state.choice(possible_concat_nodes,  # basically chose x where x is distributed as half a gaussian but it needs revise
                                                            get_truncated_normal(upp=len(
                                                                possible_concat_nodes)))  # todo change this as it is not just want you want please see with upp 3 it is very likly

                    to_connect_with_node = [to_connect_with_node]

                    if isinstance(concat_nodes, list):
                        to_connect_with_node.extend(concat_nodes)
                        _unconnected_nodes = [node for node in concat_nodes if not node.out_connections]
                        self._width = self._width - (len(_unconnected_nodes))  # controls the width #todo
                    else:
                        to_connect_with_node.append(concat_nodes)
                        if not concat_nodes.out_connections:
                            self._width += 1 #todo

                    self.layers[self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                                 input_node=to_connect_with_node,
                                                                 _computational_layer=_computational_layer,
                                                                 seed=self.random_state))
                else:
                    self.layers[self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                                 input_node=to_connect_with_node,
                                                                 _computational_layer=_computational_layer,
                                                                 seed=self.random_state))
                    self._width += 1  #todo

    def get_semantics_initial_nodes(self):
        """builds a model to return the semantics for all nodes and saves them to each node"""
        input_node = self.initial_input_node.computational_layer
        output_nodes_semantics = [node.computational_layer for node in self.layers[self.mutation_level]]  # could save some memory to not save the semantics of a flatten or concatenate layer
        model_for_semantics = keras.models.Model(inputs=input_node, outputs=output_nodes_semantics)
        semantics = model_for_semantics.predict(self.x_train)
        self.semantics_size += sum([sem.nbytes for sem in semantics]) / 1.E6
        [setattr(self.layers[self.mutation_level][idx], 'semantics', i) for idx, i in enumerate(semantics)]
        self.initial_input_node.semantics = self.x_train


    def isolated_mutation(self):
        """Mutate the network such that a graph with only one touch-point is added to the network
        returns the network, contains more than one neuron per layer """
        self.layers[self.mutation_level] = []
        mutation_node = self.random_mutation_node()  # choose a staring node
        self.build_random_layers(starting_node=mutation_node, concatenation_restricted=True)
        self.get_semantics_mutation_nodes()
        self.mutation_level += 1
        self.semantics = self.output_node.semantics
        self.fitness = self._evaluate()
        return self

    def get_semantic_inputs(self):  # todo
        """searches if any node in the current mutation level has inputs from previous semantics and returns a tuple of
        with a list of the semantic inputs resembling a keras.layer.input and the associated data to the input"""
        semantic_input_nodes = flatten([node.semantic_input_node for node in self.layers[self.mutation_level]
                                        if node.semantic_input])
        semantic_data = flatten([node.input_data for node in self.layers[self.mutation_level] if node.semantic_input])
        return semantic_input_nodes, semantic_data

    def get_semantics_mutation_nodes(self):
        """gets the semantics for all newly added nodes in this mutation"""
        semantic_input_nodes, semantic_data = self.get_semantic_inputs()  # get the input nodes and the data
        output_nodes_semantics = [node.semantics_computational_layer for node in self.layers[self.mutation_level]]
        model_for_semantics = keras.models.Model(inputs=semantic_input_nodes,
                                                 outputs=output_nodes_semantics)  # build the model_for_semantics
        semantics = model_for_semantics.predict(semantic_data)
        self.semantics_size += sum([sem.nbytes for sem in semantics]) / 1.E6
        [setattr(self.layers[self.mutation_level][idx], 'semantics', i) for idx, i in enumerate(semantics)]

    def random_mutation_node(self):
        """:returns any Node"""
        all_cl_nodes = self.get_all_nodes()
        mutation_node = self.random_state.choice([node for node in all_cl_nodes if len(node.output_shape) > 2])
        return mutation_node

    def get_all_nodes(self):
        """ :returns list of all Nodes"""
        all_nodes = [item for sublist in self.layers.values() for item in sublist]
        all_nodes.append(self.initial_input_node)
        return all_nodes

    def _get_model(self):
        """ builds the mode"""
        self.model = keras.models.Model(inputs=self.initial_input_node.computational_layer,
                                        outputs=self.output_node.computational_layer)

    def build_dense(self):
        """Builds the dense layer part
        for the moment it is static but could easily be adapted to work as a SLM"""
        kernel_initializer = keras.initializers.RandomNormal(seed=self.random_state.randint(sys.maxsize))
        self.layers[self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                     input_node=self.layers[self.mutation_level][-1],
                                                     _computational_layer=keras.layers.Dense(
                                                         20,
                                                         activation='relu',
                                                         kernel_initializer=kernel_initializer)))

        self.layers[self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                     input_node=self.layers[self.mutation_level][-1],
                                                     _computational_layer=keras.layers.Dense(
                                                         self.n_outputs,
                                                         activation=None,
                                                         kernel_initializer=kernel_initializer)))
        if self.mutation_level == 0:
            self.output_node = self.layers[self.mutation_level][-1]  # output for the initial network is the last node
        else:
            last_nodes = self.get_last_nodes()
            self.layers[self.mutation_level].append(Node(mutation_level=self.mutation_level,
                                                         input_node=last_nodes,
                                                         _computational_layer=keras.layers.Add(), # maybe also using average or something else?
                                                         seed=0))  # maybe not a good idea as this will remain the last node such that the impact of new mutations will be less

            self.output_node = self.layers[self.mutation_level][-1]

        self._set_model()

    def get_last_nodes(self):
        """:returns the last nodes from all mutations"""
        return [sublist[-1] for sublist in self.layers.values()]

    def _evaluate(self):
        "evaluates the network"
        return cross_entropy(predictions=self.semantics, targets=self.y_train)

    def _set_model(self):
        """builds the model containing all nodes from the main graph"""
        self.model = Model(inputs=self.initial_input_node.computational_layer,
                           outputs=self.output_node.computational_layer)

    def predict(self, data):
        return self.model.predict(data)


if __name__ == '__main__':
    Network_a = NeuralNetwork(input_shape=(5, 5, 3), n_outputs=1, seed=1)  # 3,5
    Network_a.summary()
    Network_b = NeuralNetwork(input_shape=(64, 64, 3), n_outputs=1, seed=1)  # 3,5
    Network_b  # .get_random_mutation()
    # Network_b.model_2.summary()
