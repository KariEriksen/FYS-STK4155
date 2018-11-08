import numpy as np


class CostFunction :
    def __init__(   self, 
                    function = None,
                    alpha    = None) :
        self.function = function if (function is not None) else 'mse'
        self.parseString(self.function)
        
        # Regularization parameter, taking a page out of the sklearn book,
        # to avoid messing with the reserved python keyword lambda
        self.alpha    = alpha    if (alpha    is not None) else 0.0

    def parseString(self, string) :
        if self.function == 'mse' :
            self.function    = self._mse
            self.derivative_ = self._mse_derivative 
        elif self.function == 'cross-entropy' :
            self.function    = self._cross_entropy
            self.derivative  = self._cross_entropy_derivative
        else :
            raise ValueError("Cost function <" + str(string) + "> not recognized.")

    def _mse(self, y, target) :
        return ((y - target)**2).mean() * 0.5

    def _mse_derivative(self, y, target) :
        return y - target

    def _cross_entropy(self, y, target) :
        return -np.sum(target * np.log(y) + (1-target)*np.log(1-y)) / y.shape[0]

    def _cross_entropy_derivative(self, y, target) :
        #return -np.sum(target/y - (1-target)/(1-y))
        return target - y

    def derivative(self, y, target) :
        return self.derivative_(y, target)

    def cost(self, y, target) :
        return self.__call__(y, target)

    def __call__(self, y, target) :
        return self.function(y, target)