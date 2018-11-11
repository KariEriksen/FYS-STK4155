import os
import sys
import copy
import numpy as np
import functools
import pickle
import sklearn
import collections
from sklearn import datasets
import sklearn.model_selection as skms
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
    return np.sum(Y_test.astype(int) == Y_pred.astype(int)) / len(Y_test)

def total_missed(Y_test, Y_pred):
    return np.sum(Y_test.astype(int) != Y_pred.astype(int))

def mnist() :
    np.random.seed(25251)
    
    # Steal MNIST data from sklearn.
    data = datasets.load_digits()

    # Shift images +/- 1 pixel in each direction to conjure up 5 times more 
    # training data.
    n_samples_original = 1797
    images_extended  = np.zeros(shape=(n_samples_original*4, 8, 8))
    target_extended = np.zeros(n_samples_original*4)

    for i, (shift, axis) in enumerate(zip([1,1,-1,-1],[1,2,1,2])) :
        im = np.roll(copy.deepcopy(data.images), shift, axis=axis)
        """
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
        """
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

    target_train_onehot = to_onehot(target_train)

    n_labels  = target_train_onehot.shape[1]

    nn = NeuralNetwork(inputs   = n_features,
                       outputs  = n_labels,
                       cost     = 'cross-entropy')
    nn.addLayer(activations = 'sigmoid', neurons = 100)
    nn.addLayer(activations = 'sigmoid', neurons = 50)
    nn.addLayer(activations = 'softmax', neurons = n_labels, output = True)
    
    epochs = 500
    
    nn.fit(
           x_train.T,
           target_train_onehot.T,
           #images_original.T,
           #to_onehot(target_original).T,
           batch_size           = 1000,
           epochs               = epochs,
           validation_fraction  = 0.2,
           validation_skip      = 5,
           verbose              = False,
           optimizer            = 'adam',
           lmbda                = 0.0)
    
    #nn = pickle.load(open('nn.p', 'rb'))

    y_validation = nn.predict(x_validation.T)
    y_validation = to_label(y_validation)

    acc = accuracy_score_numpy(target_validation, y_validation)

    print("Accuracy: ", acc)

    # Display up to 50 of the digits misclassified
    miss_ind = np.where(target_validation != y_validation)
    miss = x_validation[miss_ind]
    n_missed = miss.shape[0]
    random_indices = np.random.choice(np.arange(n_missed), size=min(n_missed, 50))
    
    for i, image in enumerate(miss[random_indices]):
        ax = plt.subplot(5, 5, i+1)
        plt.axis('on')
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
        fontsize = 5
        classified = 'guess: ' + str(y_validation     [miss_ind[0][random_indices[i]]])
        true       = 'true: '  + str(target_validation[miss_ind[0][random_indices[i]]])
        plt.ylabel(classified, fontsize=fontsize)
        plt.title(true , fontsize=fontsize)
        plt.imshow(image.reshape((8,8)), cmap=plt.cm.gray_r)
    vspace = 0.5
    hspace = 0.5
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.subplots_adjust(wspace=vspace, hspace=hspace)
    plt.show()


def load_ising_data() :
    L = 40 #Nr of spins 40x40

    label_filename = "data/Ising2DFM_reSample_L40_T=All_labels.pkl"
    dat_filename   = "data/Ising2DFM_reSample_L40_T=All.pkl"

    # Read in the labels
    with open(label_filename, "rb") as f:
        labels = pickle.load(f)

    # Read in the corresponding configurations
    with open(dat_filename, "rb") as f:
        data = np.unpackbits(pickle.load(f)).reshape(-1, 1600).astype("int")

    data[data     == 0] = -1

    # Set up slices of the dataset
    ordered    = slice(0     , 70000 )
    critical   = slice(70000 , 100000)
    disordered = slice(100000, 160000)


    X_train, \
    X_test,  \
    y_train, \
    y_test = skms.train_test_split(
                np.concatenate((data  [ordered], data  [disordered])),
                np.concatenate((labels[ordered], labels[disordered])),
                test_size = 0.5,
                shuffle   = True)
    X_critical = data  [critical]
    y_critical = labels[critical]

    del data, labels
    return X_train, X_test, y_train, y_test, X_critical, y_critical


def save_ising_smaller(n_samples = 5000) :
    X_train, X_test, y_train, y_test = load_ising_data()

    X_train_smaller = X_train[:n_samples]
    X_test_smaller  = X_test [:n_samples]

    y_train_smaller = y_train[:n_samples]
    y_test_smaller  = y_test [:n_samples]

    data = {'x_train':      X_train_smaller,
            'target_train': y_train_smaller,
            'x_test':       X_test_smaller,
            'target_test':  y_test_smaller}

    with open('ising_data_'+str(n_samples)+'.p', "wb") as file_handle:
        pickle.dump(data, 
                    file_handle, 
                    protocol = pickle.HIGHEST_PROTOCOL) 

    return X_train_smaller, \
           X_test_smaller , \
           y_train_smaller, \
           y_test_smaller


def load_ising_smaller(n_samples = 5000) :
    try : 
        with open('ising_data_'+str(n_samples)+'.p', "rb") as file_handle:
            data = pickle.load(file_handle)

        return data['x_train'],      \
               data['x_test'],       \
               data['target_train'], \
               data['target_test']
    except :
        X_train_smaller, \
        X_test_smaller , \
        y_train_smaller, \
        y_test_smaller = save_ising_smaller(n_samples)

        return X_train_smaller, \
               X_test_smaller , \
               y_train_smaller, \
               y_test_smaller 


def ising_classify(n_samples = 5000) :
    np.random.seed(294782)

    x_train,      \
    x_test,       \
    target_train, \
    target_test = load_ising_data()
    #target_test = load_ising_smaller(n_samples)
    n_samples = 65000
    target_train = to_onehot(target_train)

    n_labels   = target_train.shape[1]
    n_features = x_train.shape[1]

    nn = NeuralNetwork(inputs   = n_features,
                       outputs  = n_labels,
                       cost     = 'cross-entropy')
    nn.addLayer(activations = 'sigmoid', neurons = 100)
    nn.addLayer(activations = 'softmax', neurons = n_labels, output = True)
    
    epochs = int(1e6)
    
    nn.fit(x_train.T,
           target_train.T,
           batch_size           = 5000,
           epochs               = epochs,
           validation_fraction  = 0.2,
           validation_skip      = 5,
           verbose              = False,
           optimizer            = 'adam',
           lmbda                = 0.0,
           save_always          = True)
    """
    with open('nn.p', "rb") as f:
        nn = pickle.load(f)
    """
    y_test = nn.predict(x_test.T)
    y_test = np.squeeze(to_label(y_test, axis=0))
    target_test = np.squeeze(target_test)

    print("Critical accuracy: ", 
          accuracy_score_numpy(target_test,
                               y_test))

    y_train = nn.predict(x_train.T)
    y_train = np.squeeze(to_label(y_train, axis=0))
    target_train = np.squeeze(to_label(target_train, axis=1))
    
    print("Training accuracy: ", 
          accuracy_score_numpy(target_train,
                               y_train))


def accuracy_as_function_of_epochs(n_samples=5000) :
    #np.random.seed(2248782)

    x_train,      \
    x_test,       \
    target_train, \
    target_test = load_ising_smaller(n_samples)

    target_train = to_onehot(target_train)

    n_labels   = target_train.shape[1]
    n_features = x_train.shape[1]

    nn = NeuralNetwork(inputs   = n_features,
                       outputs  = n_labels,
                       cost     = 'cross-entropy',
                       silent   = True)
    nn.addLayer(activations = 'sigmoid', neurons = 100)
    nn.addLayer(activations = 'softmax', neurons = n_labels, output = True)
    
    M = 10
    epochs = 100
    critical_accuracy = np.zeros((M,epochs))
    training_accuracy = np.zeros((M,epochs))
    for it in range(M) :
        
        nn.weights = None
        nn.addLayer(activations = 'sigmoid', neurons = 100)
        nn.addLayer(activations = 'softmax', neurons = n_labels, output = True)
    
        for ep in range(epochs) :
            nn.fit(x_train.T,
                   target_train.T,
                   batch_size           = 500,
                   epochs               = 1,
                   validation_fraction  = 0.01,
                   validation_skip      = 50,
                   verbose              = False,
                   silent               = True,
                   optimizer            = 'sgd',
                   lmbda                = 0.0,
                   save_always          = True)

            acc_crit = accuracy_score_numpy(
                            np.squeeze(target_test),
                            np.squeeze(to_label(nn.predict(x_test.T), axis=0))
                        )
            acc_train = accuracy_score_numpy(
                            np.squeeze(to_label(target_train, axis=1)),
                            np.squeeze(to_label(nn.predict(x_train.T), axis=0))
                        )
            
            print("%-20.15f %-20.15f" %(acc_crit,acc_train))

            critical_accuracy[it,ep] = acc_crit
            training_accuracy[it,ep] = acc_train
        print("********************")

    plt.figure(1)
    for it in range(M) :
        plt.semilogy(np.arange(epochs), critical_accuracy[it,:])
    plt.rc('text', usetex=True)
    plt.xlabel(r'Epochs', fontsize=10)
    plt.ylabel(r'Validation accuracy', fontsize=10)
    plt.xlim((0,epochs))
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'epoch-validation-accuracy-adam.png'), transparent=True, bbox_inches='tight')

    plt.figure(2)
    for it in range(M) :
        plt.semilogy(np.arange(epochs), training_accuracy[it,:])
    plt.rc('text', usetex=True)
    plt.xlabel(r'Epochs', fontsize=10)
    plt.ylabel(r'Training accuracy', fontsize=10)
    plt.xlim((0,epochs))
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'epoch-training-accuracy-adam.png'), transparent=True, bbox_inches='tight')
    plt.show()

def loss_and_accuracy(n_samples=5000) :
    np.random.seed(1234)

    x_train,      \
    x_test,       \
    target_train, \
    target_test = load_ising_smaller(n_samples)

    target_train = to_onehot(target_train)

    n_labels   = target_train.shape[1]
    n_features = x_train.shape[1]

    nn = NeuralNetwork(inputs   = n_features,
                       outputs  = n_labels,
                       cost     = 'cross-entropy',
                       silent   = True)
    nn.addLayer(activations = 'sigmoid', neurons = 100)
    nn.addLayer(activations = 'softmax', neurons = n_labels, output = True)
    
    epochs = 1000
    critical_accuracy = np.zeros((epochs))
    training_accuracy = np.zeros((epochs))
    
    for ep in range(epochs) :
        nn.fit(x_train.T,
               target_train.T,
               batch_size           = 2000,
               epochs               = 2,
               validation_fraction  = 0.01,
               validation_skip      = 50,
               verbose              = False,
               silent               = True,
               optimizer            = 'sgd',
               lmbda                = 0.0,
               save_always          = True)

        acc_crit = accuracy_score_numpy(
                        np.squeeze(target_test),
                        np.squeeze(to_label(nn.predict(x_test.T), axis=0))
                    )
        acc_train = accuracy_score_numpy(
                        np.squeeze(to_label(target_train, axis=1)),
                        np.squeeze(to_label(nn.predict(x_train.T), axis=0))
                    )
        
        print("%-20.15f %-20.15f" %(acc_crit,acc_train))

        critical_accuracy[ep] = acc_crit
        training_accuracy[ep] = acc_train
    print("********************")

    plt.figure(1)
    plt.semilogy(np.arange(epochs), critical_accuracy)
    plt.rc('text', usetex=True)
    plt.xlabel(r'Epochs', fontsize=10)
    plt.ylabel(r'Validation accuracy', fontsize=10)
    plt.xlim((0,epochs))
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'accuracy-adam.png'), transparent=True, bbox_inches='tight')

    np.savetxt('acc_crit', acc_crit)
    np.savetxt('acc_train', acc_train)

def check_accuracy(fileName) :
    nn = pickle.load(open(fileName, 'rb'))
    
    x_train,      \
    x_test,       \
    target_train, \
    target_test,  \
    x_crit,       \
    target_crit = load_ising_data()

    pred_test  = np.squeeze(to_label(nn.predict(x_test.T),  axis=0))
    pred_train = np.squeeze(to_label(nn.predict(x_train.T), axis=0))
    pred_crit  = np.squeeze(to_label(nn.predict(x_crit.T),  axis=0))

    acc_crit = accuracy_score_numpy(
                    np.squeeze(target_crit),
                    pred_crit
                )
    acc_train = accuracy_score_numpy(
                    np.squeeze(target_train),
                    pred_train
                )
    miss_crit = total_missed(
                    np.squeeze(target_crit),
                    pred_crit
                )
    miss_train = total_missed(
                    np.squeeze(target_train),
                    pred_train
                )

    print(target_test.shape)
    print(target_train.shape)
    print(target_crit.shape)
    print("Accuracy (critical): %20.15f    total missed: %-20d" % (acc_crit,  miss_crit))
    print("Accuracy (training): %20.15f    total missed: %-20d" % (acc_train, miss_train))


def trim_nn_file(fileName) :
    nn = pickle.load(open(fileName, 'rb'))
    
    nn.a = None
    nn.validation_loss_improving = None
    nn.grad = None
    nn.validation_loss = None
    nn.x_validation = None
    nn.x_train = None
    nn.R2 = None
    nn.target_validation = None
    nn.target_train = None
    nn.training_loss = None
    nn.loss = None
    nn.d_weights = None
    nn.d_biases = None
    nn.delta = None
    nn.best_param = None
    nn.change = None
    nn.param_save = None
    
    nn.m = None
    nn.v = None
    nn.mh = None
    nn.vh = None
    nn.param = None

    nn.first_feedforward = True
    nn.first_backprop = True
    nn.adam_initialized = False

    variables = nn.__dict__
    for v in variables :
        vv = variables[str(v)]
        print(v, sys.getsizeof(vv), end=" ")
        if isinstance(vv, collections.Iterable) :
            for i,vvv in enumerate(vv) :
                if i < 10 :
                    if hasattr(vvv, 'nbytes') :
                        print(vvv.nbytes, end=" ")
                    else :
                        print(sys.getsizeof(vvv), end=" ")
                if i == 10 :
                    print("total i: ", i)
                if i > 10 :
                    print("  ", i , end="\r")
        print("")

    pickle.dump(nn, open('nn.p', 'wb'))


def plot_csv() :
    crit_acc  = np.genfromtxt('crit_acc_20k.txt',  delimiter=',')
    train_acc = np.genfromtxt('train_acc_20k.txt', delimiter=',')

    epochs = np.linspace(0,2*len(crit_acc)-1,len(crit_acc))

    fontsize = 10
    plt.rc('text', usetex=True)
    plt.semilogy(epochs, 1.0 - crit_acc,  label=r'validation samples')
    plt.semilogy(epochs, 1.0 - train_acc, label=r'training samples')

    plt.xlabel(r'Epochs',   fontsize=fontsize)
    plt.ylabel(r'$1-\text{accuracy}$', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xlim((0,max(epochs)))
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'accuracy-20k.png'), transparent=True, bbox_inches='tight')
    plt.show()



    


if __name__ == '__main__':
    #mnist()
    #ising_classify(20000)
    #accuracy_as_function_of_epochs(5000)
    #loss_and_accuracy(20000)

    #ising_classify()
    check_accuracy('nn_classify20k_2k_epochs.p')

    #trim_nn_file('nn_65k.p')

    #plot_csv()







