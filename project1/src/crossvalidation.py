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

		#print(x)
		#print("***************************************")
		#split x in k folds
		M = x.shape[0]//k

		for i in range(k):

			x_k = x[i*M:(i+1)*M,:]
			y_k = y[i*M:(i+1)*M]
			
			if (i == 0) :
				#print("** i=0 **")
				X      = self.designMatrix.getMatrix(x[:(i+1)*M,:])
				y_fold =  y[:(i+1)*M]
				#print(x_k)
				#print()
				#print(x[(i+1)*M:,:])
			
			elif (i == k-1) :
				#print("** i=k-1 **")
				x_k = x[i*M:,:]
				y_k = y[i*M:]
				X = self.designMatrix.getMatrix(x[:i*M,:])
				y_fold = y[:i*M]
				#print(x_k)
				#print()
				#print(x[:i*M,:])


			else :
				#print("** middle **")
				mask   = np.array([True for i in range(x.shape[0])])
				mask[i*M:(i+1)*M] = False
				X 	   = self.designMatrix.getMatrix(x[mask,:])
				y_fold = y[mask]
				#print(x_k)
				#print()
				#print(x[mask,:])

			beta = self.leastSquares.fit(X,y_fold)
			X_k  = self.designMatrix.getMatrix(x_k)
			y_predict = np.dot(X_k,beta)
			MSE_k = np.sum((y_k-y_predict)**2)/M 
			print(MSE_k)



if __name__ == '__main__':
	
	degree = 5
	designMatrix = DesignMatrix('polynomial2D', degree)
	leastSquares = LeastSquares(backend='manual')
	crossvalidation = CrossValidation(leastSquares, designMatrix)
	
	N = int(1000)
	x = np.random.rand(N)
	y = np.random.rand(N)
	x_data = np.zeros(shape=(N,2))
	x_data[:,0] = x
	x_data[:,1] = y
	y_data = np.zeros(shape=(N))

	@jit(nopython=True, cache=True)
	def computeFrankeValues(x_data, y) :    
		N = x_data.shape[0]
		for i in range(N) :
			y[i] = franke(x_data[i,0], x_data[i,1])

	computeFrankeValues(x_data, y_data)
	crossvalidation.kFoldCrossValidation(x_data,y_data)