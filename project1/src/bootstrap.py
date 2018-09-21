import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt

# Add the src/ directory to the python path so we can import the code 
# we need to test directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from leastSquares import LeastSquares
from designMatrix import DesignMatrix


class Bootstrap :
	"""Perform bootstrap resampling of ordinary least sqaures fits
	
	Originally proposed by B. Efron in 1979. For the original paper, see 
	https://projecteuclid.org/euclid.aos/1176344552. 
	"""

	def __init__(self, leastSquares, designMatrix) :
		"""Bootstrap class constructor

		Parameters
		----------
		leastSquares : LeastSquares
			A LeastSquares class instance used for performing the fits
		designMatrix : DesignMatrix
			A DesignMatrix class instance used for setting up the design 
			matrices necessary for the performing the fits
		"""
		self.leastSquares = leastSquares
		self.designMatrix = designMatrix


	def resample(self, x, y, k) :
		"""Perform bootstrap resampling

		Performs a k fold bootstrapping of the data supplied in the parameter
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
			the number of times to perform the bootstrapping scheme
		"""

		# If a fit has not yet been done, we perform a minimal one just to
		# obtain the number of beta parameters in the model. This is an awful
		# way to do this, but I am too lazy to add the necessary logic to 
		# make it smart right now.
		if not self.leastSquares.fitDone :
			if self.designMatrix.model == 'polynomial2D' :
				X = self.designMatrix.getMatrix(x[:2,:])
			else :
				X = self.designMatrix.getMatrix(x[:2])
			self.leastSquares.fit(X, y[:2])
		self.betaMatrix = np.zeros(shape=(self.leastSquares.beta.size, k))

		for i in range(k) :
			N  = x.shape[0]

			# Pick random indices from the input data x, this corresponds to
			# picking N values from x with replacement.
			indices = np.random.randint(0, N, N)

			yi = y[indices]
			if self.designMatrix.model == 'polynomial2D' :
				xi = x[indices,:]
			else :
				xi = x[indices]
			Xi = self.designMatrix.getMatrix(xi)
			self.betaMatrix[:,i] = self.leastSquares.fit(Xi, yi)

			# Simple progress bar
			pp = math.floor(40*i/k)
			print("Resampling %6d/%6d : [" % (i,k) + pp*"="+(40-pp)*" "+"]\r",end="")
		print(17*" "+12*" "+40*" "+"\r", end="")

		self.beta 		  = np.average(self.betaMatrix, axis=1)
		self.betaVariance = np.var 	  (self.betaMatrix, axis=1)

		self.leastSquares.beta = self.beta
		X = self.designMatrix.getMatrix(x)
		self.leastSquares.X = X
		self.leastSquares.y = y
