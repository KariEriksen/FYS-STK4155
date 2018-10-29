import numpy as np
import sys


class Activation :
    def __init__(   self, 
                    function = None,
                    alpha    = None) :
        self.function = function if (function is not None) else 'sigmoid'
        self.alpha    = alpha    if (alpha    is not None) else 0.1
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
            return self._sigmoid
        elif string == 'tanh' : 
            return self._tanh
        elif string == 'relu' :
            return self._relu
        elif string == 'leakyrelu' or string == 'leaky_relu' :
            return self._leakyrelu
        elif string == 'identity' :
            return self._identity
        else :
            raise ValueError("Unrecognized activation function <" + str(string) + ">.")

    def _sigmoid(self, x) :
        return 1.0 / (np.exp(-x) + 1.0)

    def _tanh(self, x) :
        return np.tanh(x)

    def _relu(self, x) :
        np.clip(x, a_min = 0.0, a_max = sys.float_info.max, out = x)
        return x

    def _leakyrelu(self, x) :
        return (x >= 0.0) * x + (x < 0.0) * (self.alpha * x)

    def _identity(self, x) :
        return x

    def __call__(self, x) :
        return self.function(x)