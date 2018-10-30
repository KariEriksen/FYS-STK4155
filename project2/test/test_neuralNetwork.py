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


def test_neuralNetwork_network( silent = False) :
    # Lets set up a sci-kit learn neural network and copy over the weights 
    # and biases to our network, verify that the two give the exact same 
    # result.

    from sklearn.neural_network import MLPRegressor

    X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [0, 2, 4, 6, 8, 10]
    mlp = MLPRegressor( solver              = 'sgd', 
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
                        activations = 'relu',
                        silent      = silent )
    nn.addLayer()
    nn.addLayer()
    nn.addOutputLayer(activations = 'identity')

    W_nn = nn.weights
    b_nn = nn.biases

    for i in range(len(W_nn)) :
        W_nn[i] = W_skl[i]
    for i in range(len(b_nn)) :
        b_nn[i] = np.expand_dims(b_skl[i], axis=1)

    X_test = np.array([[1.2857], [9.2508255], [-5.25255], [3.251095]])

    output_skl = mlp.predict(X_test)
    output_nn  = np.squeeze(nn(X_test.T))

    if not silent :
        print("%20.15f %20.15f %20.15f %20.15f" % (*output_skl,))
        print("%20.15f %20.15f %20.15f %20.15f" % (*output_nn,))
    assert output_nn == pytest.approx(output_skl)

    return nn, mlp


def test_neuralNetwork_backpropagation() :
    # We re-use the test_neuralNetwork_network networks and this time check
    # that the computed backpropagation derivatives are equal.

    from sklearn.neural_network import MLPRegressor

    X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [0, 2, 4, 6, 8, 10]
    mlp = MLPRegressor( solver              = 'sgd', 
                        alpha               = 0.0,
                        hidden_layer_sizes  = (3, 3), 
                        random_state        = 1,
                        activation          = 'logistic')
    # Force sklearn to set up all the matrices by fitting a data set.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp.fit(X,y)
    
    # Throw away all the fitted values, randomize W and b matrices.
    
    np.random.seed(18)
    for i, coeff in enumerate(mlp.coefs_) :
        mlp.coefs_[i]      = np.random.normal(size=coeff.shape)
    for i, bias in enumerate(mlp.intercepts_) :
        mlp.intercepts_[i] = np.random.normal(size=bias.shape)


    W_skl = mlp.coefs_
    b_skl = mlp.intercepts_

    nn = NeuralNetwork( inputs      = 1,
                        outputs     = 1,
                        layers      = 3,
                        neurons     = 3,
                        activations = 'sigmoid',
                        silent      = False )
    nn.addLayer()
    nn.addLayer()
    nn.addOutputLayer(activations = 'identity')
    nn.weights = W_skl
    for i, b in enumerate(b_skl) :
        nn.biases[i]  = np.expand_dims(b, axis=1)
 
    # From the sklearn source, we need to set up some lists to use the _backprop
    # function in MLPRegressor, see:
    #
    #    https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/neural_network/multilayer_perceptron.py#L355
    #
    # ========================================================================
    # Initialize lists
    X = np.array([[1.125982598]])#, [2.5], [3.5], [4.5]])
    y = np.array([ 8.29289285])#,   5.0,   7.0,   5.0])
    mlp.predict(X)
    n_samples, n_features = X.shape
    batch_size = n_samples
    hidden_layer_sizes = mlp.hidden_layer_sizes
    # Make sure self.hidden_layer_sizes is a list
    if not hasattr(hidden_layer_sizes, "__iter__"):
        hidden_layer_sizes = [hidden_layer_sizes]
    hidden_layer_sizes = list(hidden_layer_sizes)
    layer_units = ([n_features] + hidden_layer_sizes + [mlp.n_outputs_])
    activations = [X]
    activations.extend(np.empty((batch_size, n_fan_out)) 
                       for n_fan_out in layer_units[1:])
    deltas = [np.empty_like(a_layer) for a_layer in activations]
    coef_grads = [np.empty((n_fan_in_, n_fan_out_)) 
                  for n_fan_in_, n_fan_out_ in zip(layer_units[:-1],
                                                   layer_units[1:])]
    intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in
                       layer_units[1:]]
    # ========================================================================
    activations = mlp._forward_pass(activations)
    loss, coef_grads, intercept_grads = mlp._backprop(
            X, y, activations, deltas, coef_grads, intercept_grads)
    
    yhat = nn(X)
    nn.backpropagation(yhat, y)
    """

    print("coefs_:")
    for c in mlp.coefs_ :
        print(c)
    print("intercepts_:")
    for i in mlp.intercepts_ :
        print(i)
    print(".-.--.-.-.-.-.-.-.-.-.-.")
    print("loss:", loss)

    print("coef_grads:")
    for grad in coef_grads :
        print(grad)


    print("intercept_grads: ")
    for grad in intercept_grads :
        print(grad)

    yhat = nn(X)
    nn.backpropagation(yhat, y)
    print(nn.delta[-1])
    print("====================")
    for a in activations :
        print(a)
    print(" ")
    for a in nn.a :
        print(a.T)
    print("--------------------")
    for w in nn.weights :
        print(w.shape)
    print("")
    for w in mlp.coefs_ :
        print(w.shape)
    print("......................")
    print(nn.biases[0].shape)
    print(mlp.out_activation_)
    """
    print("nn.deltas: ")
    for d in nn.delta :
        print(d)
    print("")
    print("mlp.intercept_grads: ")
    for d in intercept_grads :
        print(d)

    print("====================")
    print("activations::::")
    for a in activations :
        print(a)
    print(" ")
    for a in nn.a :
        print(a.T)
    print("--------------------")

    for i, d_bias in enumerate(nn.d_biases) :
        assert np.squeeze(d_bias) == pytest.approx(np.squeeze(intercept_grads[i]))

    print("====-----=====-----=====----====")
    print("D_WEIGHT")
    for i, d_weight in enumerate(nn.d_weights) :
        #assert np.squeeze(d_weight) == pytest.approx(np.squeeze(coef_grads[i]))
        print(d_weight)
        print(coef_grads[i])
        print("")

if __name__ == '__main__':
    test_neuralNetwork_backpropagation()









