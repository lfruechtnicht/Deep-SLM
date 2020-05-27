from scipy.optimize.lbfgsb import fmin_l_bfgs_b
from sklearn.neural_network._base import LOSS_FUNCTIONS, DERIVATIVES, \
    ACTIVATIONS
from sklearn.utils.extmath import safe_sparse_dot

import numpy as np


class LBFGS:
    
    def __init__(self, max_iter=50, verbose=False):
        self.max_iter = max_iter
        self.verbose = verbose
        #=======================================================================
        # self.activation = 'tanh'
        #=======================================================================
        self.activation = 'relu'
        self.out_activation_ = 'softmax'
        self.tol = 1e-4
        self.alpha = 0
        
    def fit(self, X, y, activations, deltas, coef_grads, intercept_grads, layer_units, random_state, coef_init=None, intercept_init=None, init_bound=1, fixed_weighted_input=None):
        self.n_iter_ = 0
        self.n_outputs_ = y.shape[1]

        self.coefs_ = []
        self.intercepts_ = []
        # Compute the number of layers
        self.n_layers_ = len(layer_units)
        
        if coef_init is None and intercept_init is None:
            for i in range(self.n_layers_ - 1):
                coef_init = random_state.uniform(-init_bound, init_bound, (layer_units[i], layer_units[i + 1]))
                intercept_init = random_state.uniform(-init_bound, init_bound, layer_units[i + 1])
                self.coefs_.append(coef_init)
                self.intercepts_.append(intercept_init)
        else:
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)
        
        self.fixed_weighted_input = fixed_weighted_input
        
        self._fit_inner(X, y, activations, deltas, coef_grads, intercept_grads, layer_units)
        
        return self.coefs_, self.intercepts_
    
    def _fit_inner(self, X, y, activations, deltas, coef_grads, intercept_grads, layer_units):
    
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0
    
        # Save sizes and indices of coefficients for faster unpacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]
    
            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end
    
        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end
    
        # Run LBFGS
        packed_coef_inter = self._pack(self.coefs_, self.intercepts_)
    
        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1
    
        optimal_parameters, self.loss_, d = fmin_l_bfgs_b(x0=packed_coef_inter, func=self._loss_grad_lbfgs, maxfun=self.max_iter, iprint=iprint, pgtol=self.tol, args=(X, y, activations, deltas, coef_grads, intercept_grads))
        
        self._unpack(optimal_parameters)
    
    def _forward_pass(self, activations):
        hidden_activation = ACTIVATIONS[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(activations[i],
                                                 self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]

            # For the hidden layers
            if (i + 1) != (self.n_layers_ - 1):
                activations[i + 1] = hidden_activation(activations[i + 1])

        # For the last layer
        output_activation = ACTIVATIONS[self.out_activation_]
        
        if self.fixed_weighted_input is not None:
            activations[i + 1] += self.fixed_weighted_input
        
        activations[i + 1] = output_activation(activations[i + 1])

        return activations
    
    def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads):
        n_samples = y.shape[0]
        # n_samples = X.shape[0]
        
        # Forward propagate
        activations = self._forward_pass(activations)
        
        # Get loss
        loss_func_name = 'log_loss'
        loss = LOSS_FUNCTIONS[loss_func_name](y, activations[-1])
        
        # Backward propagate
        last = self.n_layers_ - 2
        
        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = activations[-1] - y
        
        # Compute gradient for the last layer
        coef_grads, intercept_grads = self._compute_loss_grad(last, n_samples, activations, deltas, coef_grads, intercept_grads)
        
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative = DERIVATIVES[self.activation]
            inplace_derivative(activations[i], deltas[i - 1])
        
            coef_grads, intercept_grads = self._compute_loss_grad(i - 1, n_samples, activations, deltas, coef_grads, intercept_grads)
        
        return loss, coef_grads, intercept_grads
    
    def _compute_loss_grad(self, layer, n_samples, activations, deltas,
                           coef_grads, intercept_grads):
        coef_grads[layer] = safe_sparse_dot(activations[layer].T,
                                            deltas[layer])
        coef_grads[layer] += (self.alpha * self.coefs_[layer])
        coef_grads[layer] /= n_samples

        intercept_grads[layer] = np.mean(deltas[layer], 0)

        return coef_grads, intercept_grads
    
    def _loss_grad_lbfgs(self, packed_coef_inter, X, y, activations, deltas, coef_grads, intercept_grads):
        self._unpack(packed_coef_inter)
        loss, coef_grads, intercept_grads = self._backprop(X, y, activations, deltas, coef_grads, intercept_grads)
        self.n_iter_ += 1
        grad = self._pack(coef_grads, intercept_grads)
        return loss, grad
    
    def _pack(self, coefs_, intercepts_):
        """Pack the parameters into a single vector."""
        return np.hstack([l.ravel() for l in coefs_ + intercepts_])
    
    def _unpack(self, packed_parameters):
        """Extract the coefficients and intercepts from packed_parameters."""
        for i in range(self.n_layers_ - 1):
            start, end, shape = self._coef_indptr[i]
            self.coefs_[i] = np.reshape(packed_parameters[start:end], shape)
        
            start, end = self._intercept_indptr[i]
            self.intercepts_[i] = packed_parameters[start:end]
