import numpy as np
import sys
import os
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
			X = self.designMatrix.getMatrix(x[:2])
			self.leastSquares.fit(X, y[:2])
		self.betaMatrix = np.zeros(shape=(self.leastSquares.beta.size, k))

		if __name__ == '__main__':
			plt.figure()
			indices = np.argsort(x)
			plt.plot(x[indices], y[indices], 'k-')

		for i in range(k) :
			N  = x.size

			# Pick random indices from the input data x, this corresponds to
			# picking N values from x with replacement.
			indices = np.random.randint(0, N, N)

			yi = y[indices]
			xi = x[indices]
			Xi = self.designMatrix.getMatrix(xi)
			self.betaMatrix[:,i] = self.leastSquares.fit(Xi, yi)
			
			if __name__ == '__main__':
				yiHat = self.leastSquares.predict()
				indices = np.argsort(xi)
				plt.plot(xi[indices], yiHat[indices], 'y--')

		self.beta 		  = np.average(self.betaMatrix, axis=1)
		self.betaVariance = np.var 	  (self.betaMatrix, axis=1)

		print(self.beta)
		print(self.betaVariance)

		if __name__ == '__main__':
			X = self.designMatrix.getMatrix(x)
			self.leastSquares.beta = self.beta
			self.leastSquares.X = X
			yHat = self.leastSquares.predict()
			indices = np.argsort(x)

			plt.plot(x[indices], yHat[indices], 'r-')
			plt.show()




			



if __name__ == '__main__':
	ls = LeastSquares()
	dm = DesignMatrix('polynomial', 3)
	bs = Bootstrap(ls, dm)
	x = np.random.rand(100)*5
	y = x**2 + x - 0.01*x**3 + np.cos(6*x/np.pi) - np.exp(-0.5*x)
	bs.resample(x,y,3)