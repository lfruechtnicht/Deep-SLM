

class Neuron:
    """Base class for different types of neural_network_components that compose a neural network.

    Warning: this class should not be used directly. Use derived classes instead.
    """

    def __init__(self, bias, semantics):
        self.bias = bias
        self.semantics = semantics

    def __repr__(self):
        return "Neuron"

    def clean_semantics(self):
        """Resets the semantics."""
        self.semantics = None
        
    def free(self):
        self.semantics = None

    def get_semantics(self):
        """Returns current semantics of the neuron."""
        return self.semantics
    
    def print_semantics_range(self):
        #=======================================================================
        # print("\n\t[Debug] Semantics range:", self.semantics.shape)
        # print("\tSemantics [min, mean, max]: [%.3f, %.3f, %.3f]" % (self.semantics.min(), self.semantics.mean(), self.semantics.max()))
        #=======================================================================
        
        #=======================================================================
        # print('\n\tMin = %.3f' % (self.semantics.min()))
        # import numpy as np
        # percentiles = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
        # for i in percentiles:
        #     print("\tPercentile %d = %.3f" % (i, np.percentile(self.semantics, i)))
        # print('\tMax = %.3f' % (self.semantics.max()))
        #=======================================================================
        
        pass
