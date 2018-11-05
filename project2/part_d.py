import os
import sys
import numpy as np
import functools
import matplotlib.pyplot as plt

# Add the src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'project1', 'src'))

from ising          import Ising
from neuralNetwork  import NeuralNetwork



def train_net_predict_energy(L=20, N=1000) :
    ising  = Ising(L,N)
    X, y   = ising.generateDesignMatrix1D()
    n_samples, n_features = X.shape

    nn = NeuralNetwork( inputs          = n_features,
                        neurons         = n_features,
                        outputs         = 1,
                        activations     = 'sigmoid',
                        cost            = 'mse',
                        silent          = False)
    nn.addLayer(neurons=n_features)
    nn.addOutputLayer(activations = 'identity')
    nn.fit( X.T, 
            y,
            shuffle             = True,
            batch_size          = 400,
            validation_fraction = 0.2,
            learning_rate       = 0.001,
            verbose             = False,
            silent              = False,
            epochs              = 1000,
            validation_skip     = 100,
            optimizer           = 'adam')
    yHat = nn.predict(X.T)
    ind  = np.argsort(y)
    yHat = yHat.T

    plt.plot(y[ind], 'ko')
    plt.plot(yHat[ind], 'r.')
    plt.show()






if __name__ == '__main__':
    train_net_predict_energy()