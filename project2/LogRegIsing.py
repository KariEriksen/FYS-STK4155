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
from scipy.special import expit


def gradientAscent(X,y,Lambda,eta=5e-7,max_iters=150,tolerance=1e-4):
    
    #Initialize beta-parameters
    beta   = np.zeros(X.shape[1])
    norm   = 100
    
    for i in range(0,max_iters):
        
        z = np.dot(X, beta)
        p = expit(z)

        gradient = np.dot(X.T,y-p) - Lambda*beta
        beta    += eta*gradient
        norm     = np.linalg.norm(gradient)
        
        if(i%1000 == 0):
            print(norm)

        if(norm < tolerance):
            print("Gradient ascent converged to given precision in %d iterations" % i)
            break
    
    return beta, norm

np.random.seed(12)
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
ordered    = slice(0     , 5000 )
critical   = slice(70000 , 100000)
disordered = slice(100000, 104900)


X_train, X_test, y_train, y_test = skms.train_test_split(
    np.concatenate((data[ordered], data[disordered])),
    np.concatenate((labels[ordered], labels[disordered])),
    test_size=0.5
)

X_critical = data[critical]
y_critical = labels[critical]

del data, labels


Lambda = 1

logreg = skl.LogisticRegression(fit_intercept=True,C=1.0/Lambda,tol=1e-8)
logreg.fit(X_train,y_train)
beta_scikit = np.zeros(X_train.shape[1]+1)
beta_scikit[0] = logreg.intercept_
beta_scikit[1:] = logreg.coef_[0]
print(beta_scikit[0:10])

# check accuracy
train_accuracy    = logreg.score(X_train,y_train)
test_accuracy     = logreg.score(X_test,y_test)
crit_accuracy = logreg.score(X_critical,y_critical)
print("Training accuracy (skl): %.6f" % train_accuracy)
print("Test accuracy     (skl): %.6f" % test_accuracy)
print("Critical accuracy (skl): %g" % crit_accuracy)

sys.exit(1)
#Add intercept column
X_train    = np.c_[np.ones(X_train.shape[0]), X_train]
X_test     = np.c_[np.ones(X_test.shape[0]), X_test]
X_critical = np.c_[np.ones(X_critical.shape[0]), X_critical]
print(X_train.shape)

beta,norm  = gradientAscent(X_train,y_train,Lambda=Lambda,max_iters=100000,tolerance=1e-4)

p_predict  = expit(np.dot(X_train,beta))
p_test     = expit(np.dot(X_test,beta))
p_critical = expit(np.dot(X_critical,beta))

train_accuracy = ( (p_predict  > 0.5) == y_train     ).mean()
test_accuracy  = ( (p_test     > 0.5) == y_test      ).mean()
crit_accuracy  = ( (p_critical > 0.5) == y_critical  ).mean()

#print("Lambda: %g" % lmbda)
print("norm(gradC): %g" % norm)
print("Training accuracy: %.6f" % train_accuracy)
print("Test accuracy: %.6f" % test_accuracy)
print("Critical accuracy: %g" % crit_accuracy)



print(beta[0])
print(beta[1:4])


"""
# plot accuracy against regularisation strength
plt.plot(lmbdas,train_accuracy,'*-b',label='Train')
plt.plot(lmbdas,test_accuracy,'*-r',label='Test')
plt.plot(lmbdas,crit_accuracy,'*-g',label='Critical')

plt.xlabel('$\\lambda$')
plt.ylabel('$\\mathrm{accuracy}$')

plt.grid()
plt.legend()
plt.savefig("figures/logReg_acc_vs_regstrength.png")
plt.show()
"""