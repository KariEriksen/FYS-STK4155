import sys
import os
import numpy as np
import warnings


# Add the project2/src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from activation import Activation


class NeuralNetwork :

    def __init__(   self,
                    inputs      = None,
                    outputs     = None,
                    layers      = None,
                    neurons     = None,
                    activations = None) :

        self.inputs         = inputs
        self.outputs        = outputs
        self.layers         = layers
        self.neurons        = neurons
        self.activations    = activations

        self.weights        = None
        self.biases         = None


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
                        outputs     = None) :
        self.addLayer(  outputs = outputs,
                        output  = True)

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

            print(  "Adding input layer with " + str(neurons) + " neurons "  +
                    "using " + str(activations) + " activations.")
            W = np.random.uniform(-1.0, 1.0, size=(inputs, neurons))
            b = np.zeros(shape=(neurons,1))
            f = Activation(function = activations, alpha = alpha)

            self.weights        = [W]
            self.biases         = [b]
            self.activations    = [f]
            
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

            print(  "Adding output layer with " + str(outputs) + " outputs.")
            previousLayerNeurons = self.weights[-1].shape[1]
            W = np.random.uniform(-1.0, 1.0, size=(previousLayerNeurons, outputs))
            b = np.zeros(shape=(outputs,1))
            f = np.vectorize(lambda x : x)
            
            self.weights    .append(W)
            self.biases     .append(b)
            self.activations.append(f)

        else :
            print(  "Adding layer with " + str(neurons) + " neurons using "  +
                    activations + " activations.")
            previousLayerNeurons = self.weights[-1].shape[1]
            W = np.random.uniform(-1.0, 1.0, size=(previousLayerNeurons, neurons))
            b = np.zeros(shape=(neurons,1))
            f = Activation(function = activations, alpha = alpha)

            self.weights    .append(W)
            self.biases     .append(b)
            self.activations.append(f)


    def layer(self, x, layer_number) :
        i = layer_number

        W = self.weights[i]
        b = self.biases[i]
        f = self.activations[i]

        return f(np.dot(W.T,x) + b)


    def __call__(self, x) :
        return self.network(x)


    def network(self, x) :
        for i in range(len(self.weights)) :
            x = self.layer(x, i)
        return x







