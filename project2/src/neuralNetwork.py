import sys
import os
import warnings
import copy
import time
import numpy as np
import sklearn
import pickle
from sklearn import metrics
from math import ceil

# Add the project2/src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from activation     import Activation
from costFunction   import CostFunction


class NeuralNetwork :

    def __init__(self,
                 inputs      = None,
                 outputs     = None,
                 layers      = None,
                 neurons     = None,
                 activations = None,
                 cost        = 'mse',
                 silent      = False) :

        self.inputs         = inputs
        self.outputs        = outputs
        self.layers         = layers
        self.neurons        = neurons
        self.activations    = activations
        self.silent         = silent

        self.weights        = None
        self.biases         = None

        self.cost           = CostFunction(cost)

        self.first_feedforward = True
        self.first_backprop    = True
        self.adam_initialized  = False

    def set(self,
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


    def initializeWeight(self,
                         n_in,
                         n_out,
                         activations) :
        
        # Xavier initializations (http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).
        if activations == 'sigmoid' :
            r = np.sqrt(6.0 / (n_in + n_out))
            return np.random.uniform(-r, r, size=(n_in, n_out))
        elif activations == 'tanh' :
            r = 4.0 * np.sqrt(6.0 / (n_in + n_out))
            return np.random.uniform(-r, r, size=(n_in, n_out))


        # He initializations (https://arxiv.org/pdf/1502.01852.pdf).
        elif activations == 'relu' or activations == 'leaky_relu' or activations == 'elu' :
            return np.random.normal(size=(n_in, n_out)) * np.sqrt(2.0 / n_in)


        elif activations == 'identity' :
            return np.random.normal(size=(n_in, n_out))


    def addOutputLayer(self,
                       outputs     = None,
                       activations = None) :

        self.addLayer(outputs     = outputs,
                      output      = True,
                      activations = activations)

    def addLayer(self,
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
            #W = np.random.uniform(-1.0, 1.0, size=(inputs, neurons))
            #b = np.random.uniform(-0.1, 0.1, size=(neurons,1))
            W = self.initializeWeight(inputs, neurons, activations)
            b = np.zeros(shape=(neurons,1))
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
            #W = np.random.uniform(-1.0, 1.0, size=(previousLayerNeurons, outputs))
            #b = np.random.uniform(-0.1, 0.1, size=(outputs,1))
            W = self.initializeWeight(previousLayerNeurons, outputs, activations)
            b = np.zeros(shape=(outputs,1))
            f = Activation(function = activations, alpha = alpha)
            
            self.weights.append(W)
            self.biases .append(b)
            self.act    .append(f)

        else :
            if not self.silent :
                print(  "Adding layer with " + str(neurons) + " neurons using "  +
                        str(activations) + " activations.")
            previousLayerNeurons = self.weights[-1].shape[1]
            #W = np.random.uniform(-1.0, 1.0, size=(previousLayerNeurons, neurons))
            #b = np.random.uniform(-0.1, 0.1, size=(neurons,1))
            W = self.initializeWeight(previousLayerNeurons, neurons, activations)
            b = np.zeros(shape=(neurons,1))
            f = Activation(function = activations, alpha = alpha)

            self.weights.append(W)
            self.biases .append(b)
            self.act    .append(f)


    def layer(self, x, layer_number) :
        i = layer_number

        W = self.weights[i]
        b = self.biases[i]
        f = self.act[i]

        self.a[i+1] = f(np.dot(W.T, x) + b)

        return self.a[i+1]


    def __call__(self, x) :
        return self.network(x)

    def predict(self, x) :
        return self.network(x)

    def forward_pass(self, x) :
        return self.network(x)

    def network(self, x) :
        if self.first_feedforward :
            self.a = [None]*(len(self.weights)+1)
            self.first_feedforward = False

        self.n_features, self.n_samples = x.shape

        # First layer
        self.a[0] = x
        self.a[1] = self.act[0](np.dot(self.weights[0].T, x) + self.biases[0])
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

        self.delta[-1]      = (  self.cost.derivative(y, target) 
                               * self.act[-1].derivative(self.a[-1].T) )
        self.d_weights[-1]  = np.dot(self.a[-2], self.delta[-1]) / self.n_samples
        self.d_biases[-1]   = np.mean(self.delta[-1], axis = 0, keepdims = True).T
        
        # Iterate backwards through the layers
        for i in range(2, len(self.weights)+1) :
            self.delta[-i]      = (  np.dot(self.delta[-i+1], self.weights[-i+1].T)
                                   * self.act[-i].derivative(self.a[-i].T) )
            self.d_weights[-i]  = np.dot(self.a[-i-1], self.delta[-i]) / self.n_samples
            self.d_biases[-i]   = np.mean(self.delta[-i], axis = 0, keepdims = True).T


    def fit(self,
            x,
            target,
            shuffle             = True,
            learning_rate       = 0.001,
            epochs              = 200,
            batch_size          = 200,
            validation_fraction = 0.1,
            validation_skip     = 10,
            verbose             = False,
            silent              = False,
            optimizer           = 'sgd') :

        self.learning_rate = learning_rate
        self.best_loss  = None
        self.bestEpoch  = None
        self.best_param = None

        if optimizer == 'adam' :
            self.initializeAdam()
            self.optimizer = self.adam
        elif optimizer == 'sgd' :
            self.optimizer = self.sgd
        else :
            raise ValueError(   "The optimizer " + str(optimizer) + " is not supported.")

        self.n_features, self.n_samples = x.shape

        if not (self.n_features == self.inputs) :
            raise ValueError(   "The number of features in the input data does not equal " +
                                "the number of network *inputs*.") 
        self.n_validation       = int(round(validation_fraction*self.n_samples))
        self.x_validation       = x     [:,:self.n_validation]
        if target.ndim == 1:
            target = target.reshape((1, -1))
        self.target_validation  = target[:,:self.n_validation]
        self.x_train        = x     [:,self.n_validation+1:]
        self.target_train   = target[:,self.n_validation+1:]

        self.batch_size = min(batch_size, self.n_samples - self.n_validation) 
        if batch_size > self.batch_size :
            warning_string = ("The specified batch_size (" + str(batch_size) + ") is larger than "      +
                              "the available data set size (" + str(self.n_samples-self.n_validation)   +
                              ") after reserving " + str(self.n_validation) + " samples for "           +
                              "validation. Using batch_size = " + str(self.n_samples-self.n_validation)) 
            warnings.warn(warning_string)

        
        validation_it        = 0
        self.validation_loss = np.zeros(int(ceil(epochs / validation_skip))+1)
        self.training_loss   = np.zeros(epochs)
        self.R2              = np.zeros_like(self.validation_loss)
        self.R2 *= np.nan
        self.validation_loss_improving = np.zeros_like(self.validation_loss)
        self.validation_loss_improving *= np.nan

        if not shuffle :
            raise NotImplementedError("~")
        else :

            if not silent :
                #         ep   t/b   t/e    t    rt   bcost vcost
                print(" %-8s %-20s %-20s %-15s %-20s %-15s %-15s " % ("Epoch", 
                                                                      "Time per batch",
                                                                      "Time this epoch",
                                                                      "Elapsed time",
                                                                      "Remaining time",
                                                                      "Batch cost",
                                                                      "Validation cost"))
            self.batches_per_epoch = int(ceil(self.x_train.shape[1] / self.batch_size))
            
            start_time = time.time()

            for epoch in range(epochs) :    
                epoch_start_time   = time.time()
                batch_time_average = 0

                epoch_loss = 0

                for batch in range(self.batches_per_epoch) :
                    batch_start_time = time.time()
                    x_batch, target_batch   = sklearn.utils.shuffle(self.x_train.T, 
                                                                    self.target_train.T, 
                                                                    n_samples = self.batch_size)
                    y_batch = self.forward_pass(x_batch.T)
                    self.backpropagation(y_batch.T, target_batch)
                    batch_loss = self.cost(y_batch.T, target_batch)
                    epoch_loss += batch_loss

                    self.optimizer()

                    batch_time_average += batch_start_time - time.time()
                
                self.training_loss[epoch] = epoch_loss
                batch_time_average /= float(self.batches_per_epoch)
                epoch_time          = time.time() - epoch_start_time

                if verbose or (epoch % 5 == 0) :
                    #       ep      t/b   t/e    t    rt     bcost   vcost
                    print(" %5d    %-20s %-20s %-15.3s %-20s %-15.5g %-15s " % (epoch, 
                                                                                "",
                                                                                "",
                                                                                "",
                                                                                "",
                                                                                epoch_loss,
                                                                                ""))

                # Every validation_skip epochs, test against the validation set.
                if epoch % validation_skip == 0 or (epoch == epochs-1):
                    y_validation = self.forward_pass(self.x_validation)
                    self.validation_loss[validation_it] = self.cost(y_validation, self.target_validation)
                    self.loss = self.validation_loss[validation_it]
                    self.validation_loss_improving[validation_it] = self.best_loss

                    save = False
                    if (self.best_loss is None) or \
                       (self.best_loss > self.validation_loss[validation_it]) :

                        self.best_loss  = self.validation_loss[validation_it]
                        self.best_param = [params for params in self.weights+self.biases]
                        self.bestEpoch  = epoch
                        self.R2[validation_it] = metrics.r2_score(np.squeeze(self.target_validation), np.squeeze(y_validation))
                        self.validation_loss_improving[validation_it] = self.validation_loss[validation_it]
                        pickle.dump(self, open('nn.p', 'wb'))
                        save = True

                    validation_it += 1


                    if not silent :
                        #       ep      t/b   t/e    t    rt   bcost vcost
                        print(" %5s    %-20s %-20s %-15.3s %-20s %-15s %-15.5g %5s" % ("", 
                                                                                    "",
                                                                                    "",
                                                                                    "",
                                                                                    "",
                                                                                    "",
                                                                                    self.validation_loss[validation_it-1],
                                                                                    "ckpt" if save else ""))
        self.validation_loss_improving[-1] = self.best_loss

        # Finished fitting, set the weights / biases to the best found throughout 
        # training, i.e. the ones which resulted in the lowest validation cost.
        self.weights = [w for w in self.best_param[:len(self.weights)]]
        self.biases  = [b for b in self.best_param[len(self.weights):]]

    def sgd(self) :
        for i, d_w in enumerate(self.d_weights) :
            self.weights[i] -= self.learning_rate * d_w
        for i, d_b in enumerate(self.d_biases) :
            self.biases[i]  -= self.learning_rate * d_b


    def initializeAdam(self) :
        if not self.adam_initialized :
            self.t = 0
            self.learning_rate_init = self.learning_rate

            self.param = self.weights + self.biases
            
            # Biased first moment estimates
            self.m = [np.zeros_like(p) for p in self.param]
        
            # Biased second raw moment estimates
            self.v = [np.zeros_like(p) for p in self.param]

            # Bias-corrected first moment estimates
            self.mh = [np.zeros_like(p) for p in self.param]

            # Bias-corrected second raw moment estimates
            self.vh = [np.zeros_like(p) for p in self.param]

            self.adam_initialized = True


    def adam(self) :
        beta1   = 0.9
        beta2   = 0.999
        epsilon = 1e-8
        self.t += 1
        t       = self.t

        # Gradients
        self.grad = self.d_weights + self.d_biases

        #      
        # m  ← β  m + (1 − β )  ∇C
        #       1           1
        self.m = [beta1 * m + (1 - beta1) * g 
                  for m,g in zip(self.m, self.grad)]

        #                        2
        # v  ← β  v + (1 − β ) ∇C
        #       2           2
        self.v = [beta2 * v + (1 - beta2) * g**2
                  for v,g in zip(self.v, self.grad)]

        #             ________
        #            /     t
        #           / 1 - β 
        #          √       2
        # α  ←  α  -----------
        #                  t
        #             1 - β
        #                  1
        self.learning_rate = self.learning_rate_init * np.sqrt(1 - beta2**t) / (1 - beta1**t)

        #                
        #                         α m
        # params ←  params -  ------------
        #                       ___
        #                      √ v  + ϵ
        change = [- self.learning_rate * m / (np.sqrt(v) + epsilon) 
                  for m,v in zip(self.m, self.v)]
        self.change = change

        self.weights = [w + dw for w, dw in zip(self.weights, change[:len(self.weights)])]
        self.biases  = [b + db for b, db in zip(self.biases,  change[len(self.weights):])]





