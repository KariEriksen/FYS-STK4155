import os
import sys
import numpy as np
import functools
import pickle
import matplotlib.pyplot as plt

# Add the src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'project1', 'src'))

from ising          import Ising
from neuralNetwork  import NeuralNetwork



def train_net_predict_energy(L = 10, N = 5000) :
    ising = Ising(L, N)
    X, y  = ising.generateTrainingData1D()
    y    /= L
    n_samples, n_features = X.shape

    nn = NeuralNetwork( inputs          = L,
                        neurons         = L*L,
                        outputs         = 1,
                        activations     = 'sigmoid',
                        cost            = 'mse',
                        silent          = False)
    nn.addLayer(neurons = L*L)
    nn.addLayer(neurons = L*L)
    nn.addOutputLayer(activations = 'identity')

    validation_skip = 10
    epochs = 1000
    nn.fit( X.T, 
            y,
            shuffle             = True,
            batch_size          = 1000,
            validation_fraction = 0.2,
            learning_rate       = 0.001,
            verbose             = False,
            silent              = False,
            epochs              = epochs,
            validation_skip     = validation_skip,
            optimizer           = 'adam')
    
    # Use the net to predict the energies for the validation set.
    x_validation      = nn.x_validation
    y_validation      = nn.predict(x_validation)
    target_validation = nn.target_validation

    # Sort the targets for better visualization of the network output.
    ind = np.argsort(target_validation)
    y_validation      = np.squeeze(y_validation.T[ind])
    target_validation = np.squeeze(target_validation.T[ind])

    # We dont want to plot the discontinuities in the target.
    target_validation[np.where(np.abs(np.diff(target_validation))>1e-5)] = np.nan

    plt.rc('text', usetex=True)
    plt.figure()
    plt.plot(target_validation, 'k--', label=r'Target')
    plt.plot(y_validation,      'r.', markersize=0.5, label=r'NN output')    
    plt.legend(fontsize=10)
    plt.xlabel(r'Validation sample', fontsize=10)
    plt.ylabel(r'$E / L$',           fontsize=10)
    #plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'nn_1d_energy_predict' + str(L) + '.png'), transparent=True, bbox_inches='tight')

    # Plot the training / validation loss during training.
    training_loss     = nn.training_loss
    validation_loss   = nn.validation_loss

    # There are more training loss values than validation loss values, lets
    # align them so the plot makes sense.
    xaxis_validation_loss       = np.zeros_like(validation_loss)
    xaxis_validation_loss[0]    = 0
    xaxis_validation_loss[1:-1] = np.arange(validation_skip,len(training_loss),validation_skip)
    xaxis_validation_loss[-1]   = len(training_loss)

    plt.figure()
    plt.semilogy(training_loss, 'r-', label=r'Training loss')
    plt.semilogy(xaxis_validation_loss, validation_loss, 'k--', label=r'Validation loss')
    plt.legend(fontsize=10)
    plt.xlabel(r'Epoch',            fontsize=10)
    plt.ylabel(r'Cost $C(\theta)$', fontsize=10)
    #plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'nn_1d_loss' + str(L) + '.png'), transparent=True, bbox_inches='tight')
    plt.show()


def load_trained_network() :
    fileName = 'nn.p'
    nn = pickle.load(open(fileName, 'rb'))
    for w in nn.weights :
        print(w)


if __name__ == '__main__':
    train_net_predict_energy()
    #load_trained_network()