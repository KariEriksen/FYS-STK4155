import os
import sys
import copy
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


def to_onehot(labels):
    assert np.squeeze(labels).ndim == 1
    n_samples    = labels.size
    n_categories = int(np.max(labels) + 1)
    one_hot      = np.zeros((n_samples, n_categories))
    one_hot[range(n_samples), labels.astype(int)] = 1
    return one_hot

def to_label(one_hot, axis=0) :
    return np.argmax(one_hot, axis=axis)

def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)


def mnist() :
    np.random.seed()
    
    # Steal MNIST data from sklearn.
    data = datasets.load_digits()

    # Shift images +/- 1 pixel in each direction to conjure up 5 times more 
    # training data.
    n_samples_original = 1797
    images_extended  = np.zeros(shape=(n_samples_original*4, 8, 8))
    target_extended = np.zeros(n_samples_original*4)

    for i, (shift, axis) in enumerate(zip([1,1,-1,-1],[1,2,1,2])) :
        im = np.roll(copy.deepcopy(data.images), shift, axis=axis)
        if shift == 1 :
            if axis == 1 :
                im[:,0,:] = 0
            else :
                im[:,:,0] = 0
        else :
            if axis == 1 :
                im[:,-1,:] = 0
            else :
                im[:,:,-1] = 0
        images_extended[i*n_samples_original:(i+1)*n_samples_original] = copy.deepcopy(im)
        target_extended[i*n_samples_original:(i+1)*n_samples_original] = copy.deepcopy(data.target)
    
    n_samples_extended, im_width, im_height = images_extended.shape
    images_extended = np.reshape(images_extended, (n_samples_extended, -1))
    n_features = images_extended.shape[1]

    # Reserve 20% of the original MNIST images for validation.
    images_original = np.reshape(data.images, (n_samples_original, -1))

    # Shuffle the images first, then we pick the first 20%.
    images_original, target_original = sklearn.utils.shuffle(images_original, 
                                                             data.target,
                                                             n_samples = n_samples_original)
    n_samples_validation = int(0.2 * n_samples_original)
    x_validation         = images_original[:n_samples_validation,:]
    target_validation    = target_original[:n_samples_validation]

    # The rest of the original, and the extended images become the training data.
    n_samples    = 5 * n_samples_original - n_samples_validation
    x_train      = np.zeros(shape=(n_samples, 64))
    target_train = np.zeros(n_samples)

    x_train[:n_samples_original*4,:] = images_extended
    x_train[n_samples_original*4:,:] = images_original[:n_samples_original-n_samples_validation:,:]

    target_train[:n_samples_original*4]  = target_extended
    target_train[n_samples_original*4:] = target_original[:n_samples_original-n_samples_validation:]

    # Make sure that none of the images are blank.
    assert not np.any(0 == np.sum(x_train,      axis=1))
    assert not np.any(0 == np.sum(x_validation, axis=1))

    # choose some random images to display
    random_indices = np.random.choice(np.arange(n_samples), size=10)
    """
    for i, image in enumerate(x_train[random_indices]):
        ax = plt.subplot(5, 5, i+1)
        plt.axis('on')
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
        plt.imshow(image.reshape((8,8)), cmap=plt.cm.gray_r)
    plt.show()
    """

    target_train_onehot = to_onehot(target_train)

    n_labels  = target_train_onehot.shape[1]

    nn = NeuralNetwork(inputs   = n_features,
                       outputs  = n_labels,
                       cost     = 'cross-entropy')
    nn.addLayer(activations = 'sigmoid', neurons = 100)
    nn.addLayer(activations = 'sigmoid', neurons = 50)
    nn.addLayer(activations = 'softmax', neurons = n_labels, output = True)
    
    epochs = 1000
    nn.fit(
           #x_train.T,
           #target_train_onehot.T,
           images_original.T,
           to_onehot(target_original).T,
           batch_size           = 200,
           epochs               = epochs,
           validation_fraction  = 0.2,
           validation_skip      = 1,
           verbose              = False,
           optimizer            = 'adam',
           lmbda                = 0.0)


    y_validation = nn.predict(x_validation.T)
    print(y_validation.shape)
    print(target_validation.shape)
    y_validation = to_label(y_validation)
    print(y_validation.shape)

    acc = accuracy_score_numpy(target_validation, y_validation)

    print("accuracy: ", acc)





if __name__ == '__main__':
    mnist()








