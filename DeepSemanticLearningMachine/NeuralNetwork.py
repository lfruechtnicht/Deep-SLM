from itertools import product
import sys

from numpy import where
from sklearn.metrics.classification import accuracy_score
from tensorflow import keras

from DeepSemanticLearningMachine.Node import InputNode, MergeNode, ConvNode, DenseNode, FlattenNode
import tensorflow as tf
from tensorflow.keras.models import Model
from utils import *

from .non_convolutional.common.neural_network_builder import NeuralNetworkBuilder
from .non_convolutional.components.input_neuron import InputNeuron
from .non_convolutional.mutation import calculate_ols
from .non_convolutional.mutation import mutate_hidden_layers


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


class NeuralNetwork(object):
    """A neural network object consists of  an input Nodes, layers and an output Node. Together they form
    the model which is a Keras computational model. The network can be mutated such that additional layers are added.
     A secondary graph layer is introduced to work with intermediary outputs to not recalculate the whole network.

     x_train np.array Training images
     y_train np.array Training labels
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
                 max_splits=5,
                 mutation_level=0,
                 pre_output_node=None,
                 output_node=None,
                 model=None,
                 semantics=None,
                 semantics_size=0,
                 fitness=None,
                 initial_input_node=None,
                 conv_layers=None,
                 non_conv_layers=None,
                 validation_data=None,
                 validation_metric=None,
                 random_ncp=None
                 ):

        self.random_state = check_random_state(seed)
        self.input_shape = input_shape
        self.n_outputs = n_outputs
        self.metric = metric
        self.layer_parameters = layer_parameters

        self.conv_parameters = conv_parameters
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
        self.validation_metric = validation_metric
        self.pre_output_node = pre_output_node
        self.output_node = output_node
        self.model = model
        self.semantics = semantics
        self.semantics_size = semantics_size
        self.fitness = fitness
        self.current_with = 1

        """ TODO: -Lukas NeuralNetwork __init__ """
        self.random_ncp = random_ncp

        self.initial_input_node = initial_input_node
        self.conv_layers = conv_layers
        self.non_conv_layers = non_conv_layers
        if self.initial_input_node is None:
            self.initial_input_node = InputNode(mutation_level=self.mutation_level, input_shape=self.input_shape)
            self.conv_layers = {self.mutation_level: []}
            self.non_conv_layers = {self.mutation_level: []}
            self.pre_output_node = {self.mutation_level: None}  # default arguments arguments
            self.initialize_network()

    def __repr__(self):
        """ TODO: -Lukas __repr__ """
        y_pred_as_labels = self.semantics.argmax(axis=1)
        y_train_as_labels = self.y_train.argmax(axis=1)
        print('Global accuracy: %.5f%%' % (accuracy_score(y_train_as_labels, y_pred_as_labels) * 100))
        if self.random_ncp:
            pred = self.semantics.clip(0, 1)
            for i in range(len(self.random_ncp.output_layer)):
                y_train = self.y_train[:, i]
                y_pred_train = where(pred[:, i] >= 0.5, 1, 0)
                print('\tAccuracy for neuron %d: %.5f%%' % (i + 1, accuracy_score(y_train, y_pred_train) * 100))

        return str(f"NeuralNetwork, {self.metric.name}: {self.fitness:.4f}, Mutations: {self.mutation_level}, "
                   f"MemoUsage: {self.semantics_size:.2f}mb")

    def __copy__(self):
        return self._copy()

    def copy(self):
        # print("Number of Input_Neurons in NCP: ",len(self.random_ncp.input_layer), " Number of Neurons in the last layer of the cp: ", self.conv_layers[0][-1].output_shape[1])
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
        validation_metric = self.validation_metric
        pre_output_node = self.pre_output_node
        output_node = self.output_node
        semantics = self.semantics
        semantics_size = self.semantics_size
        initial_input_node = self.initial_input_node
        conv_layers = self.conv_layers.copy()
        non_conv_layers = self.non_conv_layers.copy()
        fitness = self.fitness
        random_ncp = NeuralNetworkBuilder.clone_neural_network(self.random_ncp)
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
                             conv_layers=conv_layers,
                             non_conv_layers=non_conv_layers,
                             validation_data=validation_data,
                             validation_metric=validation_metric,
                             random_ncp=random_ncp)

    def summary(self):
        return self.model.summary()

    def initialize_network(self):
        """Builds the first network and sets semantics and determines fitness"""
        self.build_random_layers()
        # self.get_semantics_initial_nodes()
        self.semantics = self.output_node.semantics
        self.fitness = self._evaluate()

    def build_random_layers(self):
        self.build_random_cp()
        self.build_random_ncp()

    def mutate_random_layers(self, mutate_conv):

        if mutate_conv:
            self.mutate_random_cp()
        """ TODO: -Lukas build_random_layers """
        self.mutate_random_ncp(mutate_conv)
        # self.build_fixed_dense()

    def build_random_cp(self):

        # setting initial values
        building_cp = True
        free_end_nodes = [self.initial_input_node]

        while building_cp:

            if self.current_with < self.max_width_cl:
                non_flattened_nodes = [node for node in self.get_all_conv_nodes() if len(node.output_shape) == 4]
                random_node = self.random_state.choice(non_flattened_nodes)  # one input to the next node
                if random_node.out_connections:  # the with of the network gets bigger:
                    self.current_with += 1
            else:
                non_flattened_nodes = [node for node in free_end_nodes if len(node.output_shape) == 4]
                random_node = self.random_state.choice(non_flattened_nodes)  # one input to the next node

            only_flatten = random_node.depth >= self.max_depth_cl or random_node.only_flatten_test()

            if only_flatten:  # if the node has reached the max_depth it can only be connected with a flatten node
                self.conv_layers[self.mutation_level].append(FlattenNode(mutation_level=self.mutation_level,
                                                                         input_node=random_node))
            else:  # everything is possible:
                all_nodes = self.get_all_conv_nodes()  # all possible nodes of the network of the layer type
                all_nodes.remove(random_node)
                possible_concat_nodes = [node for node in all_nodes  # need to match the outer node in shape
                                         if len(node.output_shape) > 2
                                         and random_node.output_shape[1] == node.output_shape[1]
                                         and random_node.output_shape[2] == node.output_shape[2]]

                if possible_concat_nodes:  # if there are suitable nodes the current node could be merged with:
                    node = self.random_state.choice([
                        # "Flatten",
                        None,
                        "Concat"
                    ])
                    if node == "Concat":  # build the concatenation node
                        concatenation_nodes = self.random_state.choice(possible_concat_nodes,
                                                                       max(1, self.random_state.choice(
                                                                           range(len(possible_concat_nodes)))),
                                                                       replace=False).tolist()

                        _unconnected_nodes = [node for node in concatenation_nodes if not node.out_connections]
                        self.current_with = self.current_with - (len(_unconnected_nodes))  # if concat with nodes
                        concatenation_nodes.append(random_node)
                        self.conv_layers[self.mutation_level].append(
                            MergeNode(mutation_level=self.mutation_level,
                                      input_node=concatenation_nodes,
                                      _computational_layer=keras.layers.Concatenate()))

                    elif node is None:  # build the random conv node:
                        self.conv_layers[self.mutation_level].append(ConvNode(mutation_level=self.mutation_level,
                                                                              conv_parameters=self.conv_parameters,
                                                                              pool_parameters=self.pool_parameters,
                                                                              input_node=random_node,
                                                                              random_state=self.random_state))

                    else:  # build the flatten node:
                        self.conv_layers[self.mutation_level].append(FlattenNode(mutation_level=self.mutation_level,
                                                                                 input_node=random_node))

                else:  # there are no suitable nodes to connect with only normal nodes can be added at this stage:
                    node = self.random_state.choice([
                        # "Flatten",
                        None])

                    if node is None:
                        self.conv_layers[self.mutation_level].append(ConvNode(mutation_level=self.mutation_level,
                                                                              conv_parameters=self.conv_parameters,
                                                                              pool_parameters=self.pool_parameters,
                                                                              input_node=random_node,
                                                                              random_state=self.random_state))
                    else:
                        self.conv_layers[self.mutation_level].append(FlattenNode(mutation_level=self.mutation_level,
                                                                                 input_node=random_node))

            building_cp, free_end_nodes = self.check_conv_done()

        if len(free_end_nodes) > 1:  # concatenate all parallel layers to have one input for the dense layers
            self.conv_layers[self.mutation_level].append(MergeNode(mutation_level=self.mutation_level,
                                                                   input_node=free_end_nodes,
                                                                   _computational_layer=keras.layers.Concatenate()))
        self.current_with = 1
        self.get_semantics_initial_nodes()

    def check_conv_done(self):
        free_end_nodes = [node for node in self.conv_layers[self.mutation_level] if not node.out_connections]
        if not all(len(node.output_shape) == 2 for node in free_end_nodes):  # if not all nodes are flat
            # evaluates if output tensor shape is flattened or not
            building_cl = True  # here the nodes will be a input node after random choice
        else:
            building_cl = False  # here they will be finally concatenated and represent a Bottleneck:
        return building_cl, free_end_nodes

    def mutate_random_cp(self):

        # setting initial values
        building_cp = True
        random_node = self.random_mutation_node()
        warm_start = True

        while building_cp:
            if not warm_start:
                if self.current_with < self.max_width_cl:
                    non_flattened_nodes = [node for node in self.get_all_conv_nodes() if len(node.output_shape) == 4]
                    random_node = self.random_state.choice(non_flattened_nodes)  # one input to the next node
                    if random_node.out_connections:  # the with of the network gets bigger:
                        self.current_with += 1
                else:
                    non_flattened_nodes = [node for node in free_end_nodes if len(node.output_shape) == 4]
                    random_node = self.random_state.choice(non_flattened_nodes)  # one input to the next node

            only_flatten = random_node.depth >= self.max_depth_cl or random_node.only_flatten_test()

            if only_flatten:  # if the node has reached the max_depth it can only be connected with a flatten node
                self.conv_layers[self.mutation_level].append(FlattenNode(mutation_level=self.mutation_level,
                                                                         input_node=random_node))
            else:  # everything is possible:
                all_nodes = self.get_all_conv_nodes()  # all possible nodes of the network of the layer type
                all_nodes.remove(random_node)
                possible_concat_nodes = [node for node in all_nodes  # need to match the outer node in shape
                                         if len(node.output_shape) > 2
                                         and random_node.output_shape[1] == node.output_shape[1]
                                         and random_node.output_shape[2] == node.output_shape[2]]

                if possible_concat_nodes:  # if there are suitable nodes the current node could be merged with:
                    node = self.random_state.choice(["Flatten", None, "Concat"])
                    if node == "Concat":  # build the concatenation node
                        concatenation_nodes = self.random_state.choice(possible_concat_nodes,
                                                                       max(1, self.random_state.choice(
                                                                           range(len(possible_concat_nodes)))),
                                                                       replace=False).tolist()

                        _unconnected_nodes = [node for node in concatenation_nodes if not node.out_connections]
                        self.current_with = self.current_with - (len(_unconnected_nodes))  # if concat with nodes
                        concatenation_nodes.append(random_node)
                        self.conv_layers[self.mutation_level].append(
                            MergeNode(mutation_level=self.mutation_level,
                                      input_node=concatenation_nodes,
                                      _computational_layer=keras.layers.Concatenate()))

                    elif node is None:  # build the random conv node:
                        self.conv_layers[self.mutation_level].append(ConvNode(mutation_level=self.mutation_level,
                                                                              conv_parameters=self.conv_parameters,
                                                                              pool_parameters=self.pool_parameters,
                                                                              input_node=random_node,
                                                                              random_state=self.random_state))

                    else:  # build the flatten node:
                        self.conv_layers[self.mutation_level].append(FlattenNode(mutation_level=self.mutation_level,
                                                                                 input_node=random_node))

                else:  # there are no suitable nodes to connect with only normal nodes can be added at this stage:
                    node = self.random_state.choice(["Flatten", None])

                    if node is None:
                        self.conv_layers[self.mutation_level].append(ConvNode(mutation_level=self.mutation_level,
                                                                              conv_parameters=self.conv_parameters,
                                                                              pool_parameters=self.pool_parameters,
                                                                              input_node=random_node,
                                                                              random_state=self.random_state))
                    else:
                        self.conv_layers[self.mutation_level].append(FlattenNode(mutation_level=self.mutation_level,
                                                                                 input_node=random_node))

            building_cp, free_end_nodes = self.check_conv_done()

            warm_start = False

        if len(free_end_nodes) > 1:  # concatenate all parallel layers to have one input for the dense layers
            self.conv_layers[self.mutation_level].append(MergeNode(mutation_level=self.mutation_level,
                                                                   input_node=free_end_nodes,
                                                                   _computational_layer=keras.layers.Concatenate()))
        self.current_with = 1
        self.get_semantics_mutation_nodes()

    def build_random_ncp(self, feed_original_X=True):

        input_layer = list()

        if feed_original_X:
            original_X = self.x_train
            channels = original_X.shape[3]
            for i in range(channels):
                X = original_X[:, :, :, i]
                X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
                for input_data in X.T:
                    input_layer.append(InputNeuron(input_data))


        else:
            print("Network build based on random conv")
            for input_semantics in self.conv_layers[0][-1].semantics.T:
                input_layer.append(InputNeuron(input_semantics))

        """ TODO: -Lukas build_random_ncp, values from the convolutional part need to be added to the input layer """

        number_of_hidden_layers = self.random_state.randint(1, 1 + 1)
        number_of_hidden_neurons = [self.random_state.randint(5, 10 + 1) for _ in range(number_of_hidden_layers)]
        number_of_hidden_neurons[-1] = 100

        y = self.y_train
        number_of_output_neurons = y.shape[1]
        maximum_neuron_connection_weight = 0.1
        maximum_bias_weight = 0.1
        activation_function = 'identity'
        sparseness = {'sparse': True, 'minimum_sparseness': 0.75, 'maximum_sparseness': 1,
                      'prob_skip_connection': 0}
        hidden_activation_functions_ids = ['relu']
        prob_activation_hidden_layers = 1
        nn = NeuralNetworkBuilder.generate_new_neural_network(number_of_hidden_layers, number_of_hidden_neurons,
                                                              number_of_output_neurons,
                                                              maximum_neuron_connection_weight, maximum_bias_weight,
                                                              activation_function, input_layer, self.random_state,
                                                              sparseness, hidden_activation_functions_ids,
                                                              prob_activation_hidden_layers)

        calculate_ols(nn, num_last_neurons=nn.get_number_last_hidden_neurons(), target=y)
        nn.calculate_output_semantics()
        nn.clean_hidden_semantics()

        self.output_node = nn
        self.output_node.semantics = self.output_node.get_predictions()
        self.random_ncp = nn

    def mutate_random_ncp(self, mutate_conv):
        child_nn = NeuralNetworkBuilder.clone_neural_network(self.random_ncp)

        if mutate_conv:
            for input_semantics in self.conv_layers[self.mutation_level][-1].semantics.T:
                child_nn.input_layer.append(InputNeuron(input_semantics))

        learning_step = 'optimized'
        sparseness = {'sparse': True, 'minimum_sparseness': 0.75, 'maximum_sparseness': 1,
                      'prob_skip_connection': 0}

        # maximum_new_neurons_per_layer = 5
        maximum_new_neurons_per_layer = 50

        maximum_neuron_connection_weight = 0.1
        maximum_bias_weight = 0.1

        y = self.y_train
        global_preds = self.output_node.semantics
        delta_target = y - global_preds
        hidden_activation_functions_ids = ['relu']
        prob_activation_hidden_layers = 1
        child_nn = mutate_hidden_layers(child_nn, self.random_state, learning_step, sparseness,
                                        maximum_new_neurons_per_layer, maximum_neuron_connection_weight,
                                        maximum_bias_weight, delta_target, y, global_preds,
                                        hidden_activation_functions_ids, prob_activation_hidden_layers)

        child_nn.clean_hidden_semantics()  # todo ask ivo why this mus happen?

        self.output_node = child_nn
        self.output_node.semantics = self.output_node.get_predictions()
        self.random_ncp = child_nn

    def get_last_nodes(self):
        return [sublist[-1] for sublist in self.non_conv_layers.values()]

    def get_semantics_initial_nodes(self):
        """builds a model to return the semantics for all nodes and saves them to each node"""
        input_node = self.initial_input_node.computational_layer
        output_nodes = [node for node in self.conv_layers[self.mutation_level]]
        output_nodes_layer = [node.computational_layer for node in output_nodes]

        model_for_semantics = keras.models.Model(inputs=input_node, outputs=output_nodes_layer)
        semantics = model_for_semantics.predict(self.x_train)
        self.semantics_size += sum([sem.nbytes for sem in semantics]) / 1.E6

        if len(output_nodes) > 1:
            [setattr(output_nodes[idx], 'semantics', i) for idx, i in enumerate(semantics)]
        else:
            setattr(output_nodes[0], 'semantics', semantics)
        self.initial_input_node.semantics = self.x_train

    def mutation(self):
        """Mutate the network such that a graph with only one touch-point is added to the network
        returns the network, contains more than one neuron per layer """
        self.mutation_level += 1
        self.conv_layers[self.mutation_level] = []
        self.non_conv_layers[self.mutation_level] = []
        mutation_point = self.random_state.choice([True, False])  # choose a staring point
        self.mutate_random_layers(mutate_conv=mutation_point)
        self.semantics = self.output_node.semantics
        self.fitness = self._evaluate()
        return self

    def get_semantic_inputs(self):
        """searches if any node in the current mutation level has inputs from previous semantics and returns a tuple of
        with a list of the semantic inputs resembling a keras.layer.input and the associated data to the input"""
        semantic_input_nodes = flatten([node.semantic_input_node for node in self.conv_layers[self.mutation_level]
                                        if node.semantic_input])

        semantic_input_nodes.extend(flatten([node.semantic_input_node for node in
                                             self.non_conv_layers[self.mutation_level] if node.semantic_input]))

        """ TODO: -Lukas get_semantic_inputs """
        if self.random_ncp is None:
            semantic_input_nodes.append(self.pre_output_node[self.mutation_level].semantic_input_node[0])

        semantic_data = flatten([node.input_data for node in self.conv_layers[self.mutation_level]
                                 if node.semantic_input])
        semantic_data.extend(flatten([node.input_data for node in self.non_conv_layers[self.mutation_level]
                                      if node.semantic_input]))

        return semantic_input_nodes, semantic_data

    def get_semantics_mutation_nodes(self):
        """gets the semantics for all newly added nodes in this mutation"""
        semantic_input_nodes, semantic_data = self.get_semantic_inputs()  # get the input nodes and the data
        output_nodes = [node for node in self.conv_layers[self.mutation_level]]

        output_nodes_semantics = [node.semantics_computational_layer for node in output_nodes]
        model_for_semantics = keras.models.Model(inputs=semantic_input_nodes,
                                                 outputs=output_nodes_semantics)  # build the model_for_semantics
        semantics = model_for_semantics.predict(semantic_data)
        self.semantics_size += sum([sem.nbytes for sem in semantics]) / 1.E6
        """ TODO: -Lukas get_semantics_mutation_nodes """
        if len(output_nodes) > 1:
            [setattr(output_nodes[idx], 'semantics', i) for idx, i in enumerate(semantics)]
        else:
            setattr(output_nodes[0], 'semantics', semantics)

    def random_mutation_node(self):
        """:returns any Node"""
        all_nodes = self.get_all_conv_nodes()
        mutation_node = self.random_state.choice([node for node in all_nodes if len(node.output_shape) > 2])
        return mutation_node

    def get_all_nodes(self):
        """ :returns list of all Nodes"""
        all_nodes = [item for sublist in self.conv_layers.values() for item in sublist]
        all_nodes.extend([item for sublist in self.non_conv_layers.values() for item in sublist])
        all_nodes.append(self.initial_input_node)
        return all_nodes

    def get_all_conv_nodes(self):
        """ :returns list of all Nodes of conv layer_type"""  # todo
        all_nodes = [item for sublist in self.conv_layers.values() for item in sublist]
        all_nodes.append(self.initial_input_node)
        return all_nodes

    def get_all_non_conv_nodes(self, ):
        """ :returns list of all Nodes depending on the layer_type"""  # todo
        all_nodes = [item for sublist in self.non_conv_layers.values() for item in sublist]
        all_nodes.append(self.initial_input_node)
        return all_nodes

    def _evaluate(self):
        """"evaluates the network while training"""
        """ TODO: -Lukas _evaluate """
        return self.metric.evaluate(prediction=self.semantics.clip(0, 1), target=self.y_train)
        # return self.metric.evaluate(prediction=self.semantics, target=self.y_train)

    def set_model(self):
        """builds the keras model only needed for validation"""
        self.model = Model(inputs=self.initial_input_node.computational_layer,
                           outputs=self.output_node.computational_layer)

    def evaluate(self):
        if self.validation_data is None:
            raise ValueError(
                f"Need to pass Validation data, in the form of [xtest, ytest]. Cant validate on {self.validation_data}")

        """ TODO: -Lukas evaluate """
        if self.random_ncp is None:
            self.set_model()
            y_pred = self.model.predict(self.validation_data[0])
            del self.model  # not needed anymore
            """"evaluates the network on test data"""
            return self.metric.evaluate(prediction=y_pred,
                                        target=self.validation_data[1]), self.validation_metric.evaluate(
                prediction=y_pred, target=self.validation_data[1])
        else:
            pass
            # self.get_intermediate_predictions()

    def get_bottleneck_nodes(self):
        """ :returns the computational_layer for the last node in each mutation step"""
        print(type(self.conv_layers.values()))
        return [nodes[-1].computational_layer for nodes in self.conv_layers.values() if nodes]

    def get_intermediate_predictions(self, X=None):

        bottleneck_nodes = self.get_bottleneck_nodes()
        model_for_predictions = Model(inputs=self.initial_input_node.computational_layer, outputs=bottleneck_nodes)

        if X:
            self.bollleneck_predictions = model_for_predictions.predict(X)
        else:
            self.bollleneck_predictions = model_for_predictions.predict(self.validation_data[0])

    def predict(self, X=None):
        self.get_intermediate_predictions(X=X)
        y_pred = self.random_ncp.generate_predictions(X=self.bollleneck_predictions)
        return y_pred

    def _set_parameters_one_filter(self):
        filters, kernel_sizes, strides, padding, activations, pool_size, neurons = self.layer_parameters.values()
        _conv_parameters = [[1], kernel_sizes, strides, padding, activations]
        _non_conv_parameters = [[1], activations]
        return list(product(*_conv_parameters)), list(product(*_non_conv_parameters))


def mutation_map(NN):
    """Mutate the network such that a graph with only one touch-point is added to the network
    returns the network, contains more than one neuron per layer """
    NN.mutation_level += 1
    NN.conv_layers[NN.mutation_level] = []
    NN.non_conv_layers[NN.mutation_level] = []
    mutation_point = NN.random_state.choice([True, False])  # choose a staring point
    NN.mutate_random_layers(mutate_conv=mutation_point)
    NN.semantics = NN.output_node.semantics
    NN.fitness = NN._evaluate()
    return NN


if __name__ == '__main__':
    Network_a = NeuralNetwork(input_shape=(5,), n_outputs=1, seed=1)  # 3,5
    Network_a.summary()
    Network_b = NeuralNetwork(input_shape=(64, 64, 3), n_outputs=1, seed=1)  # 3,5
    Network_b  # .get_random_mutation()
    # Network_b.model_2.summary()
