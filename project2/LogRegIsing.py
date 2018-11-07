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
from scipy.optimize import minimize

def cost_function(beta,X,y):
    p    = np.dot(X, beta)
    loss = -np.sum(y*p - np.log(1+np.exp(p)))   #+ 0.5*np.dot(beta,beta)*Lambda
    return loss


def gradientDescent(X,y,Lambda,eta=1e-4,max_iters=150,tolerance=1e-4,scale=1.0):
    
    #Initialize beta-parameters
    beta    = np.random.uniform(-0.5,0.5,X.shape[1])
    beta[0] = 1
    beta /= np.linalg.norm(beta)
    beta_prev = beta.copy()

    norm    = 100
    eta_k   = 0
    
    #first step
    z = np.dot(X, beta)
    p = expit(z)
    gradient = -np.dot(X.T,y-p)/scale + Lambda*beta/scale
    beta    -= eta*gradient
    norm     = np.linalg.norm(gradient)
    
    gradient_prev = gradient.copy()
    print(norm,scale*norm)
    
    for k in range(1,max_iters):
        
        z = np.dot(X, beta)
        p = expit(z)

        gradient_prev = gradient.copy()
        gradient      = -np.dot(X.T,y-p)/scale + Lambda*beta/scale
        
        eta_k         = np.dot((beta - beta_prev),gradient-gradient_prev) / np.linalg.norm(gradient-gradient_prev)**2

        beta_prev     = beta.copy()
        beta         -= eta_k*gradient
        
        norm          = np.linalg.norm(gradient)
        
        if(k%10 == 0):
            print(norm,scale*norm)

        if(scale*norm < tolerance):
            print("Gradient Descent converged to given precision in %d iterations" % k)
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
ordered    = slice(0     , 70000 )
critical   = slice(70000 , 100000)
disordered = slice(100000, 160000)


X_train, X_test, y_train, y_test = skms.train_test_split(
    np.concatenate((data[ordered], data[disordered])),
    np.concatenate((labels[ordered], labels[disordered])),
    test_size=0.5
)

X_critical = data[critical]
y_critical = labels[critical]

del data, labels

#Add intercept column
X_train    = np.c_[np.ones(X_train.shape[0])   , X_train]
X_test     = np.c_[np.ones(X_test.shape[0])    , X_test]
X_critical = np.c_[np.ones(X_critical.shape[0]), X_critical]


# define regularisation parameter

lmbdas = np.array([1e5]) #np.logspace(-5,4,10)

# preallocate data
train_accuracy=np.zeros(lmbdas.shape,np.float64)
test_accuracy=np.zeros(lmbdas.shape,np.float64)
critical_accuracy=np.zeros(lmbdas.shape,np.float64)

"""
logreg = skl.LogisticRegression(C=C,random_state=1,verbose=0,max_iter=1E3,tol=1E-8)
logreg.fit(X_train,y_train)
beta_scikit = np.zeros(X_train.shape[1]+1)
beta_scikit[0] = logreg.intercept_
beta_scikit[1:] = logreg.coef_[0]

train_accuracy    = logreg.score(X_train,y_train)
test_accuracy     = logreg.score(X_test,y_test)
crit_accuracy     = logreg.score(X_critical,y_critical)
"""

# loop over regularisation strength
for i,Lambda in enumerate(lmbdas):
    
    beta,norm  = gradientDescent(X_train,y_train,Lambda=Lambda,eta=1e-5,max_iters=100000,tolerance=1e-4,scale=X_train.shape[0])

    p_predict  = expit(np.dot(X_train,beta))
    p_test     = expit(np.dot(X_test,beta))
    p_critical = expit(np.dot(X_critical,beta))


    # check accuracy
    print("Lambda: %g" % Lambda)

    
    print("Training accuracy (skl): %.6f" % train_accuracy)
    print("Test accuracy     (skl): %.6f" % test_accuracy)
    print("Critical accuracy (skl): %g" % critical_accuracy)
    

    train_accuracy[i] = ( (p_predict  > 0.5) == y_train     ).mean()
    test_accuracy[i]  = ( (p_test     > 0.5) == y_test      ).mean()
    critical_accuracy[i]  = ( (p_critical > 0.5) == y_critical  ).mean()

    print("Training accuracy  (GD): %.6f" % train_accuracy[i])
    print("Test accuracy      (GD): %.6f" % test_accuracy[i])
    print("Critical accuracy  (GD): %g" % critical_accuracy[i])


#np.save("train_acc",train_accuracy)
#np.save("test_acc",test_accuracy)
#np.save("crit_acc",critical_accuracy)
#print(beta_scikit[0:10])
#print(beta[0:10])

train_accuracy_old = np.load("train_acc_old.npy")
test_accuracy_old = np.load("test_acc_old.npy")
critical_accuracy_old = np.load("crit_acc_old.npy")

train_accuracy_new = np.zeros(11)
test_accuracy_new  = np.zeros(11)
critical_accuracy_new = np.zeros(11) 

train_accuracy_new[0:10] = train_accuracy_old
test_accuracy_new[0:10] = test_accuracy_old
critical_accuracy_new[0:10] = critical_accuracy_old

train_accuracy_new[10] = train_accuracy[0]
test_accuracy_new[10] = test_accuracy[0]
critical_accuracy_new[10] = critical_accuracy[0]

lmbdas = np.logspace(-5,5,11)
# plot accuracy against regularisation strength
plt.semilogx(lmbdas,train_accuracy_new,'*-b',label='Train')
plt.semilogx(lmbdas,test_accuracy_new,'*-r',label='Test')
plt.semilogx(lmbdas,critical_accuracy_new,'*-g',label='Critical')

plt.xlabel('$\\lambda$')
plt.ylabel('$\\mathrm{accuracy}$')

plt.grid()
plt.legend()
plt.savefig("figures/logReg_acc_vs_regstrength.png")
plt.show()
