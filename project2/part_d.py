import os
import sys
import numpy as np
import functools
import pickle
import sklearn
import matplotlib.pyplot as plt

# Add the src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'project1', 'src'))

from ising          import Ising
from neuralNetwork  import NeuralNetwork
from leastSquares   import LeastSquares



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

def visualize_unbalanced() :
    L = 1000
    possible_E = len(np.arange(-L,L+1,4))
    N = 2*possible_E
    print(N)
    plt.rc('text', usetex=True)
    ising = Ising(L, N)
    ising.generateTrainingData1D()
    #s = ising.states
    E = ising.E
    #M = np.sum(s,1)
    ind = np.argsort(E)
    #s = s[ind,:]
    E = E[ind]/L
    plt.plot(E,np.linspace(0,1,len(E)),'r-', label=r'Even sampling')


    ising = Ising(L,N)
    ising.generateStates1D()
    ising.computeEnergy1D()
    s_u = ising.states
    E_u = ising.E
    M_u = np.sum(s_u,1)
    E_u = np.sort(E_u)/L
    #s_u = s[ind_u,:]
    #E_u = E[ind_u]
    plt.plot(E_u, np.linspace(0,1,len(E)),'b-', label=r'Naive sampling')
    plt.axis([-1,1,0,1])
    plt.xlabel(r'Normalized energy, $E/L$',               fontsize=10)
    plt.ylabel(r'Cumulative distribution, $P(E_i\ge E)$', fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'visualize_sampling.png'), transparent=True, bbox_inches='tight')
    plt.show()


def R2_versus_lasso() :
    L = 5
    N = 10000
    training_fraction = 0.4
    ising = Ising(L, N)
    D, ry = ising.generateDesignMatrix1D()
    X, y  = ising.generateTrainingData1D()
    y    /= L

    D_train       = D [int (training_fraction*N):,:]
    ry_train      = ry[int (training_fraction*N):]
    D_validation  = D [:int(training_fraction*N),:]
    ry_validation = ry[:int(training_fraction*N)]

    lasso = LeastSquares(method='lasso', backend='skl')
    lasso.setLambda(1e-2)
    lasso.fit(D_train,ry_train)
    lasso.y = ry_validation
    lasso_R2 = sklearn.metrics.mean_squared_error(ry_validation/L,lasso.predict(D_validation)/L)

    n_samples, n_features = X.shape

    nn = NeuralNetwork( inputs          = L*L,
                        neurons         = L,
                        outputs         = 1,
                        activations     = 'identity',
                        cost            = 'mse',
                        silent          = False)
    nn.addLayer(neurons = 1)
    nn.addOutputLayer(activations = 'identity')

    validation_skip = 100
    epochs = 50000
    nn.fit( D.T, 
            ry,
            shuffle             = True,
            batch_size          = 2000,
            validation_fraction = 1-training_fraction,
            learning_rate       = 0.0001,
            verbose             = False,
            silent              = False,
            epochs              = epochs,
            validation_skip     = validation_skip,
            optimizer           = 'adam')
    
    plt.rc('text', usetex=True)
    validation_loss = nn.validation_loss_improving
    validation_ep   = np.linspace(0,epochs,len(nn.validation_loss_improving))
    plt.semilogy(validation_ep, validation_loss, 'r-', label=r'NN')
    plt.semilogy([0, epochs], np.array([lasso_R2, lasso_R2]), 'k--', label=r'Lasso')
    plt.xlabel(r'Epoch', fontsize=10)
    plt.ylabel(r'Mean squared error', fontsize=10)
    plt.legend(fontsize=10)
    plt.xlim((0,epochs))
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    if ymin > pow(10,-5) :
        ymin = pow(10,-5)
    #plt.ylim((ymin,ymax))
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'NN_compare_lasso.png'), transparent=True, bbox_inches='tight')
    #plt.show()

def visualize_beta() :
    L = 5
    nn = pickle.load(open('nn5_final.p', 'rb'))
    for w in nn.weights[:1] :
        with np.printoptions(precision=2, suppress=True) :
            print(w.reshape((-1,L)))
    #plt.imshow(nn.weights[0].reshape((L,-1)))
    #plt.show()

    N = 1000
    training_fraction = 0.4
    ising = Ising(L, N)
    D, ry = ising.generateDesignMatrix1D()
    #X, y  = ising.generateTrainingData1D()
    #y    /= L

    print(ising.states.shape)
    W = nn.weights[0].reshape((-1,L))*nn.weights[1]
    JJ = ising.J
    for i in range(10) :
        row = ising.states[i,:]
        des = D[i,:]
        E   = ry[i]
        print(row.shape)
        row = np.expand_dims(row,1)
        print("s W s:", row.T @ W.T @ row)
        print("s J s:", row.T @ JJ @ row)
        print("D w:  ", nn.weights[0].T @ des * nn.weights[1])
        print("pred: ", nn.predict(np.expand_dims(des.T,1)))
        print("E:    ", E)
        print("")

    with np.printoptions(precision=2, suppress=True) :
        for a in np.linalg.eig(W) : 
            print(a)

    print(sklearn.metrics.mean_squared_error(np.squeeze(ry), np.squeeze(nn.predict(D.T))))
    


if __name__ == '__main__':
    #train_net_predict_energy()
    #load_trained_network()
    #visualize_unbalanced()
    #R2_versus_lasso()
    visualize_beta()



