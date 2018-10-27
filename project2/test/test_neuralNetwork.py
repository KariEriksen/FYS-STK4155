import numpy as np
import scipy 
import warnings
import os 
import sys 
import pytest

# Add the project2/src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuralNetwork import NeuralNetwork


def test_neuralNetwork_init() :

    # Ensure the sizing is correctly handled when creating a new instance
    # of the network class.
    inputs  = 6
    outputs = 4
    layers  = 3
    neurons = 87

    nn = NeuralNetwork( inputs  = inputs,
                        outputs = outputs,
                        layers  = layers, 
                        neurons = neurons)
    assert nn.inputs  == inputs
    assert nn.outputs == outputs
    assert nn.layers  == layers
    assert nn.neurons == neurons


def test_neuralNetwork_set() :
    inputs  = 6
    outputs = 4
    layers  = 3
    neurons = 87

    nn = NeuralNetwork( inputs  = inputs,
                        outputs = outputs,
                        layers  = layers, 
                        neurons = neurons)
    new_inputs  = 35
    new_outputs = 23
    new_layers  = 3
    new_neurons = 10

    # Only the inputs should change
    nn.set(inputs = new_inputs)
    assert nn.inputs  == new_inputs
    assert nn.outputs == outputs
    assert nn.layers  == layers
    assert nn.neurons == neurons   

    # Only the inputs and the outputs should have changed
    nn.set(outputs = new_outputs)
    assert nn.inputs  == new_inputs
    assert nn.outputs == new_outputs
    assert nn.layers  == layers
    assert nn.neurons == neurons        

    # Inputs, outputs, and the number of layers should have changed
    nn.set(layers = new_layers)
    assert nn.inputs  == new_inputs
    assert nn.outputs == new_outputs
    assert nn.layers  == new_layers
    assert nn.neurons == neurons   

    # All the values should be new at this point
    nn.set(neurons = new_neurons)
    assert nn.inputs  == new_inputs
    assert nn.outputs == new_outputs
    assert nn.layers  == new_layers
    assert nn.neurons == new_neurons   