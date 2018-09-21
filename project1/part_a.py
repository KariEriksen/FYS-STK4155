import os
import sys
import pytest
import numba
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
    designMatrix = DesignMatrix('polynomial2D', 2)
    leastSquares = LeastSquares(backend='manual')
    bootstrap    = Bootstrap(leastSquares, designMatrix)

    N = int(1e4)
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
    
    X = designMatrix.getMatrix(x_data)
    print(X.shape)
    leastSquares.fit(X,y_data)
    y_fit = leastSquares.predict()

    if plotting :
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(x, y, y_fit, cmap='viridis', edgecolor='none');
        plt.show()

    



if __name__ == '__main__':
    part_a(plotting=True)
