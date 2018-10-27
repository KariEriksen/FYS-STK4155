import numpy as np
import scipy 
import warnings
import os 
import sys 
import pytest

# Add the project2/src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from activation import Activation

def test_activation_init() :
    # Ensure the setup is handled correctly when initializing an instance
    # of the activation class
    act = Activation()

    # Default values
    assert act.function == act._sigmoid
    assert act.alpha    == pytest.approx(0.1)

    # String to correct function conversion
    act = Activation(function = 'tanh')
    assert act.function == act._tanh

    act = Activation(function = 'relu')
    assert act.function == act._relu

    act = Activation(function = 'leakyrelu')
    assert act.function == act._leakyrelu

    act = Activation(function = 'sigmoid')
    assert act.function == act._sigmoid

    # Check wrong string error is handled correctly
    caught = False
    try :
        act = Activation(function = 'this_is_not_an_allowed_string')
    except ValueError as e :
        caught = True
    assert caught == True

    # Check alpha value specification is handled correctly
    alpha = 0.867
    act = Activation(function = 'relu', alpha = alpha)
    assert act.alpha == pytest.approx(alpha)


def test_activation_set() :
    # Default values
    act = Activation()

    # Ensure that changing default values result in changed function calls
    act.set(function = 'tanh')
    assert act.function == act._tanh

    act.set(function = 'relu')
    assert act.function == act._relu

    act.set(function = 'leakyrelu')
    assert act.function == act._leakyrelu

    # Check wrong string error is handled correctly
    caught = False
    try :
        act.set(function = 'this_is_not_an_allowed_string')
    except ValueError as e :
        caught = True
    assert caught == True

    # Ensure alpha values are set correctly
    alpha = 0.867
    act.set(alpha = alpha)
    assert act.alpha == pytest.approx(alpha)


def test_activation_functions() :
    # Ensure the correct values are calculated by the member functions

    # We compare against sklearn functions
    from sklearn.neural_network._base import tanh, relu
    from scipy.special import expit as sigmoid

    N = 100

    act = Activation(function = 'sigmoid')
    x = np.random.uniform(-10.0, 10.0, size=(N,1))
    assert act(x) == pytest.approx(sigmoid(x))

    act.set(function = 'tanh')
    x = np.random.uniform(-10.0, 10.0, size=(N,1))
    assert act(x) == pytest.approx(tanh(x))

    act.set(function = 'relu')
    x = np.random.uniform(-10.0, 10.0, size=(N,1))
    assert act(x) == pytest.approx(relu(x))

    alpha = 2.5082958
    act.set(function = 'leakyrelu', alpha = alpha)
    x = np.random.uniform(-10.0, 10.0, size=(N,1))
    assert act(x) == pytest.approx( (x>=0.0)*x + (x<0.0)*alpha*x )
    
    







