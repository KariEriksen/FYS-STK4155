import numpy as np
import scipy 
import warnings
import os 
import sys 
import pytest
import copy

# Add the project2/src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from costFunction   import CostFunction


def test_costFunction() :
    from sklearn.neural_network._base import squared_loss

    cost = CostFunction('mse')

    for i in range(1,10) :
        target = np.random.uniform(-200,200,size=(10,i))
        y      = np.random.uniform(-200,200,size=(10,i))

        assert cost(y, target) == pytest.approx(squared_loss(target, y))

    for i in range(1,10) :
        target = np.random.uniform(-200,200,size=(10,i))
        y      = np.random.uniform(-200,200,size=(10,i))

        assert cost.derivative(y, target) == pytest.approx(y - target)        







if __name__ == '__main__':
    test_costFunction()