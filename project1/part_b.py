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
from PIL import Image

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

def part_b() :

    R2           = []
    MSE          = []

    R2_noise           = []
    MSE_noise          = []
    beta_noise         = []
    betaVariance_noise = []

    noise = np.logspace(-4,-1,10)
    k = 1
    fig, ax1 = plt.subplots()
    plt.rc('text', usetex=True)


    @jit(nopython=True, cache=True)
    def computeFrankeValues(x_data, y) :    
        N = x_data.shape[0]
        for i in range(N) :
            y[i] = franke(x_data[i,0], x_data[i,1])
    ind = -1
    for lambda_ in np.linspace(0.00001, 0.01, 5) :    
        ind += 1
        MSE_noise = []

        for eta in noise :
            designMatrix = DesignMatrix('polynomial2D', 5)
            if ind == 0 :
                leastSquares = LeastSquares(backend='skl', method='ols')
            else :
                leastSquares = LeastSquares(backend='skl', method='ridge')
            
            leastSquares.setLambda(lambda_)
            bootstrap    = Bootstrap(leastSquares, designMatrix)

            N = int(1e4)
            x = np.random.rand(N)
            y = np.random.rand(N)
            x_data = np.zeros(shape=(N,2))
            x_data[:,0] = x
            x_data[:,1] = y
            y_data = np.zeros(shape=(N))
            computeFrankeValues(x_data, y_data)
            y_data_noise = y_data +  eta * np.random.standard_normal(size=N)

            bootstrap.resample(x_data, y_data_noise, k)
            

            MSE_noise.         append(leastSquares.MSE())
            R2_noise.          append(leastSquares.R2())
            beta_noise.        append(bootstrap.beta)
            betaVariance_noise.append(bootstrap.betaVariance)

            leastSquares.y = y_data
            MSE.append(leastSquares.MSE())
            R2. append(leastSquares.R2())
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        if ind == 0 :
            ax1.loglog(noise, np.array(MSE_noise), colors[ind]+'-o', markersize=2, label=r"OLS")
        else :
            ax1.loglog(noise, np.array(MSE_noise), colors[ind]+'-o', markersize=2, label=r"$\lambda=%5.3f$"%(lambda_))
        plt.ylabel(r"MSE", fontsize=10)
        plt.xlabel(r"noise scale $\eta$", fontsize=10)
        plt.subplots_adjust(left=0.2,bottom=0.2)

        #ax1.set_ylim([0.95*min(min(MSE_noise), min(R2_noise)), 1.05*(max(max(MSE_noise), max(R2_noise)))])
        
    ax1.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'MSE_ridge_noise.png'), transparent=True, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    part_b()
    



