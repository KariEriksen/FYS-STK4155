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
    assert type(nn.act[-1]) is Activation
    assert nn.act[-1].function == nn.act[-1]._sigmoid

    new_neurons     = 10
    new_activations = 'relu' 
    nn.addLayer(neurons     = new_neurons, 
                activations = new_activations)
    assert nn.weights[-1].shape == (neurons, new_neurons)
    assert nn.biases[-1] .shape == (new_neurons, 1)
    assert type(nn.act[-1]) is Activation
    assert nn.act[-1].function == nn.act[-1]._relu

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
    assert nn(x).shape          == (outputs,1)
    assert nn.network(x).shape  == (outputs,1)
    
    nn = copy.deepcopy(nn_copy3)
    assert nn(x).shape          == (new_outputs,1)
    assert nn.network(x).shape  == (new_outputs,1)


def test_neuralNetwork_network() :
    # Lets set up a sci-kit learn neural network and copy over the weights 
    # and biases to our network, verify that the two give the exact same 
    # result.

    from sklearn.neural_network import MLPRegressor

    X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [0, 2, 4, 6, 8, 10]
    mlp = MLPRegressor( solver              = 'lbfgs', 
                        alpha               = 0.0,
                        hidden_layer_sizes  = (3, 3), 
                        random_state        = 1,
                        activation          = 'relu' )
    mlp.fit(X,y)
    W_skl = mlp.coefs_
    b_skl = mlp.intercepts_

    nn = NeuralNetwork( inputs      = 1,
                        outputs     = 1,
                        layers      = 3,
                        neurons     = 3,
                        activations = 'relu' )
    nn.addLayer()
    nn.addLayer()
    nn.addOutputLayer(activations = 'identity')

    W_nn = nn.weights
    b_nn = nn.biases

    for i in range(len(W_nn)) :
        W_nn[i] = W_skl[i]
    for i in range(len(b_nn)) :
        b_nn[i] = np.expand_dims(b_skl[i], axis=1)

    X_test = np.array([[1.5], [2.5], [3.5], [4.5]])

    output_skl = mlp.predict(X_test)
    output_nn  = np.squeeze(nn.network(X_test.T))
    print(output_skl.shape)
    print(output_nn.shape)
    print("%20.15f %20.15f %20.15f %20.15f" % (*output_skl,))
    print("%20.15f %20.15f %20.15f %20.15f" % (*output_nn,))
    assert output_nn == pytest.approx(output_skl)

if __name__ == '__main__':
    test_neuralNetwork_network()








