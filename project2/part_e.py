import os
import sys
import numpy as np
import functools
import pickle
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Add the src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'project1', 'src'))

from ising          import Ising
from neuralNetwork  import NeuralNetwork
from leastSquares   import LeastSquares


def mnist() :
    np.random.seed(92898)
    # download MNIST dataset
    digits = datasets.load_digits()

    # define inputs and labels
    inputs = digits.images
    labels = digits.target

    print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
    print("labels = (n_inputs) = " + str(labels.shape))


    # flatten the image
    # the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
    n_inputs = len(inputs)
    inputs = inputs.reshape(n_inputs, -1)
    n_samples, n_features = inputs.shape
    print("X = (n_inputs, n_features) = " + str(inputs.shape))


    # choose some random images to display
    indices = np.arange(n_inputs)
    random_indices = np.random.choice(indices, size=5)

    for i, image in enumerate(digits.images[random_indices]):
        plt.subplot(1, 5, i+1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title("Label: %d" % digits.target[random_indices[i]])
    #plt.show()

    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size,
                                                        test_size=test_size)
    def to_categorical_numpy(integer_vector):
        n_inputs = len(integer_vector)
        n_categories = np.max(integer_vector) + 1
        onehot_vector = np.zeros((n_inputs, n_categories))
        onehot_vector[range(n_inputs), integer_vector] = 1
        return onehot_vector

    Y_train_onehot, Y_test_onehot = to_categorical_numpy(Y_train), to_categorical_numpy(Y_test)

    n_labels  = Y_train_onehot.shape[1]
    n_neurons = 50

    nn = NeuralNetwork(inputs   = n_features,
                       outputs  = n_labels,
                       neurons  = n_neurons,
                       cost     = 'cross-entropy')
    nn.addLayer(activations = 'sigmoid')
    nn.addLayer(activations = 'softmax', neurons = n_labels, output = True)
    
    epochs = 10000
    print(X_train.shape)
    nn.fit(X_train.T,
           Y_train_onehot.T,
           batch_size           = 800,
           epochs               = epochs,
           validation_fraction  = 0.2,
           validation_skip      = 100,
           verbose              = False,
           optimizer            = 'adam',
           lmbda                = 0.0)





if __name__ == '__main__':
    mnist()








