import numpy as np
import sys


class Activation :
    def __init__(   self, 
                    function = None,
                    alpha    = None) :
        self.derivative = None
        self.function = function if (function is not None) else 'sigmoid'
        self.alpha    = alpha    if (alpha    is not None) else 0.01
        self.name     = 'sigmoid'
        self.function = self._parseFunctionString(self.function)

    def set(    self,
                function = None,
                alpha    = None) :
        self.alpha    = alpha    if (alpha    is not None) else self.alpha
        if function is not None :
            self.function = function
            self.name     = function
            self.function = self._parseFunctionString(self.function)
    

    def _parseFunctionString(self, string) :
        self.name = string
        if string == 'sigmoid' :
            self.derivative = self._sigmoid_derivative
            return self._sigmoid
        elif string == 'tanh' : 
            self.derivative = self._tanh_derivative
            return self._tanh
        elif string == 'relu' :
            self.derivative = self._relu_derivative
            return self._relu
        elif string == 'leakyrelu' or string == 'leaky_relu' :
            self.derivative = self._leakyrelu_derivative
            return self._leakyrelu
        elif string == 'identity' :
            self.derivative = self._identity_derivative
            return self._identity
        elif string == 'elu' :
            self.derivative = self._elu_derivative
            return self._elu
        elif string == 'softmax' :
            self.derivative = self._softmax_derivative
            return self._softmax
        else :
            raise ValueError("Unrecognized activation function <" + str(string) + ">.")

    def _sigmoid(self, x) :
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_derivative(self, x) :
        return x * (1.0 - x)

    def _tanh(self, x) :
        return np.tanh(x)

    def _tanh_derivative(self, x) :
        return 1.0 - x**2

    def _relu(self, x) :
        np.clip(x, a_min = 0.0, a_max = sys.float_info.max, out = x)
        return x

    def _relu_derivative(self, x) :
        return np.heaviside(x, 0.0)

    def _leakyrelu(self, x) :
        return (x >= 0.0) * x + (x < 0.0) * (self.alpha * x)

    def _leakyrelu_derivative(self, x) :
        raise NotImplementedError("~")

    def _identity(self, x) :
        return x

    def _identity_derivative(self, x) :
        return 1.0

    def _elu(self, x) :
        neg = x<0.0
        #x[neg] = self.alpha * (np.exp(x[neg]) - 1.0)
        x[neg] = (np.exp(x[neg]) - 1.0)
        return x

    def _elu_derivative(self, x) :
        neg = x<0.0
        #x[x<0.0] = self.alpha * np.exp(x[neg])
        x[neg] = np.exp(x[neg])
        x[x >= 0.0] = 1.0

    def _softmax(self, x) :
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def _softmax_derivative(self, x) :
        return x

    def __call__(self, x) :
        return self.function(x)


