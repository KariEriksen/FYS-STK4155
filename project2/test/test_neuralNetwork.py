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

from neuralNetwork  import NeuralNetwork
from activation     import Activation


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


def test_neuralNetwork_addLayer() :
    inputs      = 6
    outputs     = 4
    layers      = 3
    neurons     = 87
    activations = 'sigmoid'

    nn = NeuralNetwork( inputs      = inputs,
                        outputs     = outputs,
                        layers      = layers, 
                        neurons     = neurons,
                        activations = activations)
    nn.addLayer()
    assert nn.weights[-1].shape == (inputs, neurons)
    assert nn.biases[-1] .shape == (neurons, 1)
    assert type(nn.activations[-1]) is Activation
    assert nn.activations[-1].function == nn.activations[-1]._sigmoid

    new_neurons     = 10
    new_activations = 'relu' 
    nn.addLayer(neurons     = new_neurons, 
                activations = new_activations)
    assert nn.weights[-1].shape == (neurons, new_neurons)
    assert nn.biases[-1] .shape == (new_neurons, 1)
    assert type(nn.activations[-1]) is Activation
    assert nn.activations[-1].function == nn.activations[-1]._relu

    nn_copy1 = copy.deepcopy(nn)
    nn_copy2 = copy.deepcopy(nn)
    nn_copy3 = copy.deepcopy(nn)
    nn_copy4 = copy.deepcopy(nn)

    nn_copy1.addOutputLayer()
    assert nn_copy1.weights[-1].shape == (new_neurons, outputs)
    assert nn_copy1.biases[-1] .shape == (outputs,1)

    nn_copy2.addLayer(output = True)
    assert nn_copy2.weights[-1].shape == (new_neurons, outputs)
    assert nn_copy2.biases[-1] .shape == (outputs,1)

    new_outputs = 24
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nn_copy3.addOutputLayer(outputs = new_outputs)
    assert nn_copy3.weights[-1].shape == (new_neurons, new_outputs)
    assert nn_copy3.biases[-1] .shape == (new_outputs,1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nn_copy4.addLayer(  outputs = new_outputs,
                            output  = True)
    assert nn_copy4.weights[-1].shape == (new_neurons, new_outputs)
    assert nn_copy4.biases[-1] .shape == (new_outputs,1)

    # Now that we constructed an entire network, check that the matrices line
    # up so that the network can be evaluated
    nn  = copy.deepcopy(nn_copy1)
    x   = np.random.uniform(-1.0, 1.0, size=(inputs,1))
    nn(x)
    nn.network(x)


if __name__ == '__main__':
    test_neuralNetwork_addLayer()







