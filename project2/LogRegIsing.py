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
	return -(1.0/len(y))*np.dot(X.T,y-p) 



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

# Set spin-down to -1
data[data == 0] = -1

"""
#Uses more memory but more compact
X_train, X_test, y_train, y_test = skms.train_test_split(
	np.concatenate((data[ordered], data[disordered])),
	np.concatenate((labels[ordered], labels[disordered])),
	test_size=0.95
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
"""

# Set up slices of the dataset
ordered_train = slice(0, 65000)
ordered_test  = slice(65000,70000)
critical = slice(70000, 100000)
disordered_train = slice(100000, 152000)
disordered_test = slice(152000,160000)

#Split data set in training and testing set
X_train_ordered    = data[ordered_train]
X_train_disordered = data[disordered_train]
y_train_ordered    = labels[ordered_train]
y_train_disordered = labels[disordered_train]

#The critical phases are left out the training
X_critical = data[critical]
y_critical = data[critical]

X_test_ordered    = data[ordered_test]
X_test_disordered = data[disordered_test]
y_test_ordered    = labels[ordered_test]
y_test_disordered = labels[disordered_test]

#vizualise()

#Initialize beta-parameters
beta = np.random.randn(L*L)
eta  = 0.005
norm = 100
Lambda = 0

for i in range(0,50):

	p_ordered    = logistic(np.dot(X_train_ordered,beta))
	p_disordered = logistic(np.dot(X_train_disordered,beta))

	gradC  = gradientCrossEntropy(X_train_disordered,p_disordered,y_train_disordered)
	gradC += gradientCrossEntropy(X_train_ordered,p_ordered,y_train_ordered)
	gradC += 2*Lambda*beta #L2 regularization

	beta   = beta - eta*gradC
	
	norm      = np.linalg.norm(gradC)
	norm_beta = np.linalg.norm(beta)
	
	print(norm, i)
