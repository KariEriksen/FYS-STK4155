import numpy as np
import pickle
import sklearn.linear_model as skl
import sklearn.metrics as skm
import sklearn.model_selection as skms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numba
import sys
import time

def vizualise():
    # set colourbar map
    cmap_args=dict(cmap='plasma_r')

    # plot states
    fig, axarr = plt.subplots(nrows=1, ncols=3)

    plt.figure(1)
    plt.subplot(311)
    plt.imshow(X_train_ordered[20001].reshape(L,L),**cmap_args)

    plt.subplot(312)
    plt.imshow(X_critical[10001].reshape(L,L),**cmap_args)

    plt.subplot(313)
    plt.imshow(X_train_disordered[50001].reshape(L,L),**cmap_args)

    plt.show()

def logistic(s):
    return np.exp(s)/(1.0+np.exp(s))

def plot_logisticFunction():
    
    s = np.linspace(-10,10,100)
    plt.plot(s,logistic(s))
    plt.show()


def crossEntropy(beta,X,y):
    
    Cbeta = 0
    
    for i in range(len(y)):
        tmp    = np.dot(X[i],beta)
        val    = y[i]*tmp - np.log(1+np.exp(tmp))
        Cbeta -= val
    
    return Cbeta

def gradientCrossEntropy(X,p,y):
    return -np.dot(X.T,y-p)/float(X.shape[0])

def gradienDescent(X_train,y_train,Lambda,eta=0.005,max_iters=50):
    
    #Initialize beta-parameters
    beta   = np.random.uniform(-0.5,0.5,L*L+1)
    beta   = beta/np.linalg.norm(beta)
    norm = 100
    
    for i in range(0,max_iters):
    
        p_hat  = logistic(np.dot(X_train,beta))
        gradC  = gradientCrossEntropy(X_train,p_hat,y_train)
        gradC += 2*Lambda*beta/X_train.shape[0]
        beta   = beta - eta*gradC
        norm   = np.linalg.norm(gradC)
        #print(norm, np.linalg.norm(beta), i)
        if(norm < 1e-5):
            break
    
    return beta, norm



np.random.seed(1)
L = 40 #Nr of spins 40x40

label_filename = "data/Ising2DFM_reSample_L40_T=All_labels.pkl"
dat_filename   = "data/Ising2DFM_reSample_L40_T=All.pkl"

# Read in the labels
with open(label_filename, "rb") as f:
    labels = pickle.load(f)

# Read in the corresponding configurations
with open(dat_filename, "rb") as f:
    data = np.unpackbits(pickle.load(f)).reshape(-1, 1600).astype("int")

data[data == 0] = -1

# Set up slices of the dataset
ordered    = slice(0, 70000)
critical   = slice(70000, 100000)
disordered = slice(100000,160000)

#Uses more memory but more compact
X_train, X_test, y_train, y_test = skms.train_test_split(
    np.concatenate((data[ordered], data[disordered])),
    np.concatenate((labels[ordered], labels[disordered])),
    test_size=0.5
)

X_critical = data[70000:100000,:]
y_critical = labels[70000:100000]

del data, labels

X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]
X_critical = np.c_[np.ones(X_critical.shape[0]), X_critical]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)




lmbdas=np.logspace(-5,5,11)

for Lambda in lmbdas:
    
    
    beta, norm = gradienDescent(X_train,y_train,Lambda,max_iters=150)
    
    p_predict = logistic(np.dot(X_train,beta))
    p_test    = logistic(np.dot(X_test,beta))
    p_critical= logistic(np.dot(X_critical,beta))
    
    train_accuracy = np.sum( (p_predict > 0.5)  == y_train )/float(X_train.shape[0])
    test_accuracy  = np.sum( (p_test > 0.5)  == y_test )/float(X_test.shape[0])
    crit_accuracy  = np.sum( (p_critical > 0.5)  == y_critical )/float(X_critical.shape[0])
    
    print("Lambda: %g" % Lambda)
    print("norm(gradC): %g" % norm)
    print("Training accuracy: %.6f" % train_accuracy)
    print("Test accuracy: %.6f" % test_accuracy)
    print("Critical accuracy: %g" % crit_accuracy)
