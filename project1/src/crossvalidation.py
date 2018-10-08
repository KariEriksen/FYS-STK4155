import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt
from numba import jit, njit

# Add the src/ directory to the python path so we can import the code 
# we need to test directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from leastSquares import LeastSquares
from designMatrix import DesignMatrix
from franke import franke

class CrossValidation:
	

	def __init__(self, leastSquares, designMatrix) :
	
		self.leastSquares = leastSquares
		self.designMatrix = designMatrix


	def kFoldCrossValidation(self,x,y,k=5):

		"""Perform k-fold cross-validation

		Performs a k fold cross-validation of the data supplied in the parameter
		x with true values supplied in the parameter y. The LeastSquares class
		instance provided in the constructor is used to perform the fit. The 
		DesignMatrix class instance provided in the constructor is used to 
		setup the design matrices necessary for the fits.

		Parameters
		----------
		x : numpy.array
			the data set, a 1D numpy array
		y : numpy.array
			the true data y values, a 1D numpy array
		k : int
			the number of folds 1 <= k <= N, where N is the number of datapoints.
		"""

		#Split input data x in k folds of size M
		M = x.shape[0]//k

		MSE_train = []
		MSE_k     = []
		R2_k      = []
		var_k     = []
		bias_k    = []

		for i in range(k):

			#x_k and y_k are the hold out data for fold k
			x_k = x[i*M:(i+1)*M,:]
			y_k = y[i*M:(i+1)*M]
			
			############################################################
			#Use the k-1 folds which are not hold out to perform the fit
			if (i == 0) :
				X_train = self.designMatrix.getMatrix(x[:(i+1)*M,:])
				y_train =  y[:(i+1)*M]
				
			elif (i == k-1) :
				x_k = x[i*M:,:]
				y_k = y[i*M:]
				X_train = self.designMatrix.getMatrix(x[:i*M,:])
				y_train = y[:i*M]
			else :
				mask   = np.array([True for i in range(x.shape[0])])
				mask[i*M:(i+1)*M] = False
				X_train = self.designMatrix.getMatrix(x[mask,:])
				y_train = y[mask]
				
			beta = self.leastSquares.fit(X_train,y_train)
			y_predict_train = np.dot(X_train,beta)
			MSE_train.append(np.sum( (y_train-y_predict_train)**2)/len(y_train))

			############################################################

			#Make prediction on the hold out set, x_k
			#and compare with the hold out y_k NOT used in the fit.
			X_k       = self.designMatrix.getMatrix(x_k)
			y_predict = np.dot(X_k,beta)

			num_data  = len(y_predict)
			MSE_test  = np.sum((y_k-y_predict)**2)/num_data 

			MSE_k.append(np.sum((y_k-y_predict)**2)/num_data)
			R2_k.append(1 - np.sum( (y_predict-y_k)**2 ) / ( np.sum ( (y_k - np.mean(y_k))**2 ) ) )
			var_k.append(np.var(y_predict))
			bias_k.append(np.sum((y_k-y_predict)/num_data))
		
		
		averages = [np.mean(MSE_k), np.mean(R2_k), np.mean(var_k), np.mean(bias_k),np.mean(MSE_train)]
		return averages


if __name__ == '__main__':
	
	Degree = 1+np.arange(20)


	train_MSE = []
	test_MSE  = []
	print("#########################################################")
	for degree in Degree:
		
		degree = int(degree)
		designMatrix = DesignMatrix('polynomial2D', degree)
		leastSquares = LeastSquares(method="ridge",backend='manual')
		Lambda = 4
		leastSquares.setLambda(Lambda)
		crossvalidation = CrossValidation(leastSquares, designMatrix)
		
		N = int(1e4)
		x = np.random.rand(N)
		y = np.random.rand(N)
		x_data = np.zeros(shape=(N,2))
		x_data[:,0] = x
		x_data[:,1] = y
		y_data = np.zeros(shape=(N))

		X = designMatrix.getMatrix(x_data)
		XT_X = np.dot(X.T,X)
		print("det(X^T*X): %g" % (np.linalg.det(XT_X+Lambda*np.eye(XT_X.shape[0]))))

		noise_strength = 0.5
		@jit(nopython=True, cache=True)
		def computeFrankeValues(x_data, y) :    
			N = x_data.shape[0]
			for i in range(N) :
				y[i] = franke(x_data[i,0], x_data[i,1]) + noise_strength*np.random.random()

		computeFrankeValues(x_data, y_data)
		
		averages = crossvalidation.kFoldCrossValidation(x_data,y_data,k=10)
		print("Maximum monomial degree in basis: %d" % degree)
		print("mean(MSE_test): %g, mean(MSE_train): %g" % (averages[0],averages[4]))
		print("mean(R2): %g" % averages[1])
		print("mean(var): %g, mean(bias)^2: %g" % (averages[2],averages[3]**2))
		print(averages[2]+averages[3]**2)
		print("#########################################################")
		test_MSE.append(averages[0])
		train_MSE.append(averages[4])

	plt.plot(Degree,np.array(test_MSE),'bo')
	plt.plot(Degree,np.array(train_MSE),'ro')
	plt.legend(["test MSE","train MSE"])
	plt.show()