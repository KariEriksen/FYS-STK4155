import sys
import os
import numpy as np
import warnings
import copy


# Add the project2/src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from activation     import Activation
from costFunction   import CostFunction


class NeuralNetwork :

    def __init__(   self,
                    inputs      = None,
                    outputs     = None,
                    layers      = None,
                    neurons     = None,
                    activations = None,
                    silent      = False) :

        self.inputs         = inputs
        self.outputs        = outputs
        self.layers         = layers
        self.neurons        = neurons
        self.activations    = activations
        self.silent         = silent

        self.weights        = None
        self.biases         = None

        self.cost           = CostFunction()

        self.first_feedforward = True
        self.first_backprop    = True

    def set(    self,
                inputs      = None,
                outputs     = None,
                layers      = None,
                neurons     = None,
                activations = None) :

        self.inputs         = inputs        if (inputs      is not None) else self.inputs
        self.outputs        = outputs       if (outputs     is not None) else self.outputs
        self.layers         = layers        if (layers      is not None) else self.layers
        self.neurons        = neurons       if (neurons     is not None) else self.neurons
        self.activations    = activations   if (activations is not None) else self.activations

    def addOutputLayer( self,
                        outputs     = None,
                        activations = None) :

        self.addLayer(  outputs     = outputs,
                        output      = True,
                        activations = activations)

    def addLayer(   self,
                    inputs      = None,
                    neurons     = None,
                    activations = None,
                    alpha       = None,
                    outputs     = None,
                    output      = False) :

        if neurons is None :
            if self.neurons is None :
                raise ValueError(   "Number of neurons is not specified. "      +
                                    "Use the NeuralNetwork class constructor, " +
                                    "the .set method, or give the number as "   +
                                    "input to this method (.addLayer).")
            else :
                neurons = self.neurons

        if activations is None :
            if self.activations is None :
                warnings.warn(  "No activation function specified, using "  +
                                "sigmoid activation for this (and all "     +
                                "subsequent layers added).")
                self.activations = 'sigmoid'
                activations = self.activations
            else :
                activations = self.activations

        if self.weights is None :
            if inputs is None :
                if self.inputs is None :
                    raise ValueError(   "The number of inputs is not specified."   +
                                        "Use the NeuralNetwork class constructor, " +
                                        "the .set method, or give the number as "   +
                                        "input to this method (.addLayer).")
                else :
                    inputs = self.inputs
            else :
                self.inputs = inputs 
            
            if not self.silent :
                print(  "Adding input layer with " + str(neurons) + " neurons "  +
                        "using " + str(activations) + " activations.")
            W = np.random.uniform(-1.0, 1.0, size=(inputs, neurons))
            b = np.random.uniform(-0.1, 0.1, size=(neurons,1))
            f = Activation(function = activations, alpha = alpha)

            self.weights = [W]
            self.biases  = [b]
            self.act     = [f]
            
        elif output == True :
            if outputs is None :
                if self.outputs is None :
                    raise ValueError(   "The number of outputs is not specified."   +
                                        "Use the NeuralNetwork class constructor, " +
                                        "the .set method, or give the number as "   +
                                        "input to this method (.addLayer / "        +
                                        ".addOutputLayer).")
                else :
                    outputs = self.outputs
            else :
                if self.outputs != outputs :
                    warnings.warn(  "The number of outputs was earlier set to "         +
                                    str(self.outputs) + ", but the value specified to " +
                                    " .addLayer / .addOutputLayer of " + str(outputs)   +
                                    " overrides this value.")
                    self.outputs = outputs

            if not self.silent :
                print(  "Adding output layer with " + str(outputs) + " outputs, "  +
                        "with " + str(activations) + " activation.")
            previousLayerNeurons = self.weights[-1].shape[1]
            W = np.random.uniform(-1.0, 1.0, size=(previousLayerNeurons, outputs))
            b = np.random.uniform(-0.1, 0.1, size=(outputs,1))
            f = Activation(function = activations, alpha = alpha)
            
            self.weights.append(W)
            self.biases .append(b)
            self.act    .append(f)

        else :
            if not self.silent :
                print(  "Adding layer with " + str(neurons) + " neurons using "  +
                        str(activations) + " activations.")
            previousLayerNeurons = self.weights[-1].shape[1]
            W = np.random.uniform(-1.0, 1.0, size=(previousLayerNeurons, neurons))
            b = np.random.uniform(-0.1, 0.1, size=(neurons,1))
            f = Activation(function = activations, alpha = alpha)

            self.weights.append(W)
            self.biases .append(b)
            self.act    .append(f)


    def layer(self, x, layer_number) :
        i = layer_number

        W = self.weights[i]
        b = self.biases[i]
        f = self.act[i]

        self.z[i+1] = np.dot(W.T, x) + b
        self.a[i+1] = f(self.z[i+1])

        return self.a[i+1]


    def __call__(self, x) :
        return self.network(x)


    def network(self, x) :
        if self.first_feedforward :
            self.z = [None]*(len(self.weights)+1)
            self.a = [None]*(len(self.weights)+1)
            self.first_feedforward = False

        # First layer
        self.a[0] = x
        self.z[0] = np.zeros(shape=x.shape)
        self.z[1] = np.dot(self.weights[0].T, x) + self.biases[0]
        self.a[1] = self.act[0](self.z[1])
        x = self.a[1]

        for i in range(1, len(self.weights)) :
            x = self.layer(x, i)
        return x


    def backpropagation(self, y, target) :
        if self.first_backprop :
            self.delta      = [None]*len(self.weights)
            self.d_weights  = copy.deepcopy(self.weights)
            self.d_biases   = copy.deepcopy(self.biases)
            self.first_backprop = False


        self.delta[-1] = (  self.cost.derivative(y, target) 
                          * self.act[-1].derivative(self.z[-1]) )







