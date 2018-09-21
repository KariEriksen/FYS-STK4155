import os
import sys
import pytest
import numba
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import functools
import time
from numba import jit

# Add the src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from franke         import franke
from designMatrix   import DesignMatrix
from leastSquares   import LeastSquares
from bootstrap      import Bootstrap

"""
def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
    return newfunc
"""

def part_a(plotting=False) :
    for degree in [2,3,4,5,6,7,8,9] :
        designMatrix = DesignMatrix('polynomial2D', degree)
        leastSquares = LeastSquares(backend='manual')
        bootstrap    = Bootstrap(leastSquares, designMatrix)

        N = int(1e5)
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
        bootstrap.resample(x_data, y_data, 100)

        if plotting :
            print("MSE: ", leastSquares.MSE())
            print("R2:  ", leastSquares.R2())
            print("Beta Variance: ", bootstrap.betaVariance.T)

            M = 100
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            x = np.linspace(0, 1, M)
            y = np.linspace(0, 1, M)
            X, Y = np.meshgrid(x,y)
            x_data = np.vstack([X.ravel(), Y.ravel()]).T
            
            # When plotting the Franke function itself, we use these lines.
            #y_data = np.zeros(shape=(x_data.data.shape[0]))
            #computeFrankeValues(x_data, y_data)

            # When plotting the linear regression model:
            XX = designMatrix.getMatrix(x_data)
            leastSquares.X = XX
            y_data = leastSquares.predict()

            Z = np.reshape(y_data.T, X.shape)

            ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,linewidth=0, antialiased=False)
            ax.set_zlim(-0.10, 1.40)
            ax.zaxis.set_major_locator(LinearLocator(5))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            ax.view_init(30, 45)

            #plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'franke.png'), transparent=True)
            plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'OLS'+str(degree)+'.png'), transparent=True)
            plt.show()

    



if __name__ == '__main__':
    part_a(plotting=True)
