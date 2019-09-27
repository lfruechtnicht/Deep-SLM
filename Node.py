from tensorflow import keras
from tensorflow.keras import layers


class Node(object):
    """ A node/layer in the directed graph which represents a Neural Network.
     Each node has exactly one input (except the input node) and at least one output node.
     """

    def __init__(self, input_node=None, is_input_node=False, is_output_node=False, computational_layer=None):
        self.input_node = input_node
        self.computational_layer = computational_layer
        self.is_input_node = is_input_node
        self.output_shape = None
        self.out_connections = []
        self.is_output_node = is_output_node  # TODO needs another evalution that output node and output connection
        self._evaluate()

    def _evaluate(self):
        """ Evaluates the Node. Either it is an Input Node and has no Input Nodes or it has input nodes and is no Input
        Node. Moreover, a Input Node must be of type Node.
        Throws ValueError: A node can only be Input Node or have an Input Node! or ValueError: A Input Node must be a
        of type Node."""
        if not self.is_input_node and self.input_node is None:
            raise ValueError("A node can only be Input Node or have an Input Node!")
        if not self.is_input_node and not isinstance(self.input_node, Node):
            raise ValueError("A Input Node must be a of type Node!")

    def _get_computational_layer(self):
        pass



if __name__ == '__main__':

    B = 1
    # A = Node(input_node=B)
    C = Node(is_input_node=True)
    D = Node(input_node=C)
