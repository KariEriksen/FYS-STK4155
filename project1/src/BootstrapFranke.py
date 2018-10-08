import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt
from numba import jit, njit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

# Add the src/ directory to the python path so we can import the code 
# we need to test directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from leastSquares import LeastSquares
from designMatrix import DesignMatrix
from franke import franke

np.random.seed(2018)

@jit(nopython=True, cache=True)
def computeFrankeValues(x_data, y, noise_strength=0.1) :    
	N = x_data.shape[0]
	for i in range(N) :
		y[i] = franke(x_data[i,0], x_data[i,1]) + np.random.normal(0, noise_strength)

leastSquares = LeastSquares(method="ridge",backend='manual')
Lambda = 1
leastSquares.setLambda(Lambda)
#crossvalidation = CrossValidation(leastSquares, designMatrix)

N = int(1e4)
x1 = np.random.rand(N)
x2 = np.random.rand(N)

X = np.zeros(shape=(N,2))
X[:,0] = x1
X[:,1] = x2

#Vector to hold y = Franke(x1,x2)
y = np.zeros(shape=(N))
noise_strength = 0.3
computeFrankeValues(X, y,noise_strength)

# Hold out some test data that is never used in training.
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.2)

X_test      = np.zeros(shape=(x1_test.shape[0],2))
X_test[:,0] = x1_test
X_test[:,1] = x2_test

max_degree = 20
Degree = []

for i in range(1,max_degree+1):
	Degree.append(i)

Bias_vec = []
Var_vec  = []
MSE_vec  = []

print("#########################################################")
for degree in Degree:
	
	degree = int(degree)
	designMatrix = DesignMatrix('polynomial2D', degree)
	
	X = designMatrix.getMatrix(X)
	XT_X = np.dot(X.T,X)
	print("det(X^T*X): %g" % (np.linalg.det(XT_X+Lambda*np.eye(XT_X.shape[0]))))

	X_test_deg = designMatrix.getMatrix(X_test)

	# The following (m x n_bootstraps) matrix holds the column vectors y_pred
	# for each bootstrap iteration.
	X_train      = np.zeros(shape=(x1_train.shape[0],2))
	n_bootstraps = 100
	y_pred 		 = np.empty((y_test.shape[0], n_bootstraps))
	
	for i in range(n_bootstraps):
		x1_, x2_, y_ = resample(x1_train,x2_train,y_train)
		X_train[:,0] = x1_
		X_train[:,1] = x2_
		X_train = designMatrix.getMatrix(X_train)

		# Evaluate the new model on the same test data each time.
		beta         = leastSquares.fit(X_train,y_)
		y_pred[:, i] = np.dot(X_test_deg,beta)
    
	y_test = y_test.reshape(len(y_test),1)
	
	MSE = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
	bias = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
	variance = np.mean( np.var(y_pred, axis=1, keepdims=True) )
	
	print("Degree: %d" % degree)
	print('Error:', MSE)
	print('Bias^2:', bias)
	print('Var:', variance)
	print('{} >= {} + {} = {}'.format(MSE, bias, variance, bias+variance))
	print("#########################################################")
	Bias_vec.append(bias)
	Var_vec.append(variance)
	MSE_vec.append(MSE)


plt.rc('text', usetex=True)
plt.figure(1)
plt.plot(Degree,np.array(MSE_vec) ,'o',label=r'MSE')
plt.plot(Degree,np.array(Bias_vec),'-|',label=r'bias${}^2$')
plt.legend()
plt.xlabel(r"Polynomial degree, $p$")
plt.ylabel(r"MSE, Bias")
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figures', "MSE_Bias_ridge_l=1.png"),transparent=True, bbox_inches='tight')

plt.figure(2)
plt.plot(Degree,np.array(Var_vec),'-*',label='variance')
plt.xlabel(r"Polynomial degree, $p$")
plt.ylabel(r"Variance")
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figures',"Var_ridge_l=1.png"),transparent=True, bbox_inches='tight')

plt.figure(3)
plt.semilogy(Degree, np.array(MSE_vec) ,'o',label=r'MSE')
plt.semilogy(Degree, np.array(Bias_vec),'-|',label=r'bias${}^2$')
plt.semilogy(Degree, np.array(Var_vec),'-*',label='variance')
plt.xlabel(r"Polynomial degree, $p$")
plt.ylabel(r"MSE, Variance, Bias")
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figures',"bias_variance_tradeoff.png"),transparent=True, bbox_inches='tight')

plt.figure(4)
fig, ax1 = plt.subplots()
ax1.plot(Degree, np.array(MSE_vec) ,'o',label=r'MSE')
ax1.plot(Degree, np.array(Bias_vec),'-|',label=r'bias${}^2$')
plt.xlabel(r"Polynomial degree, $p$", fontsize=10)
plt.ylabel(r"MSE, Bias", color='k', fontsize=10)

ax2 = ax1.twinx()
ax2.semilogy(Degree, np.array(Var_vec),'b-*',label=r'variance')
ax2.set_ylabel(r"Variance", color='b', fontsize=10)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
#plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figures',"bias_variance_tradeoff_2.png"),transparent=True, bbox_inches='tight')

plt.show()