import numpy as np
import sys


class Activation :
    def __init__(   self, 
                    function = None,
                    alpha    = None) :
        self.function = function if (function is not None) else 'sigmoid'
        self.alpha    = alpha    if (alpha    is not None) else 0.1
        self.function = self._parseFunctionString(self.function)

    def set(    self,
                function = None,
                alpha    = None) :
        self.alpha    = alpha    if (alpha    is not None) else self.alpha
        if function is not None :
            self.function = function
            self.function = self._parseFunctionString(self.function)
        

    def _parseFunctionString(self, str) :
        if str == 'sigmoid' :
            return self._sigmoid
        elif str == 'tanh' : 
            return self._tanh
        elif str == 'relu' :
            return self._relu
        elif str == 'leakyrelu' or str == 'leaky_relu' :
            return self._leakyrelu
        else :
            raise ValueError("Unrecognized activation function <" + str + ">.")

    def _sigmoid(self, x) :
        return 1.0 / (np.exp(-x) + 1.0)

    def _tanh(self, x) :
        return np.tanh(x)

    def _relu(self, x) :
        np.clip(x, a_min = 0.0, a_max = sys.float_info.max, out = x)
        return x

    def _leakyrelu(self, x) :
        return (x >= 0.0) * x + (x < 0.0) * (self.alpha * x)

    def __call__(self, x) :
        return self.function(x)