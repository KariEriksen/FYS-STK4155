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

    noise = np.linspace(0, 1.0, 50)
    k = 1
    fig, ax1 = plt.subplots()
    plt.rc('text', usetex=True)


    @jit(nopython=True, cache=True)
    def computeFrankeValues(x_data, y) :    
        N = x_data.shape[0]
        for i in range(N) :
            y[i] = franke(x_data[i,0], x_data[i,1])
    ind = -1
    for lambda_ in np.logspace(-2, 0, 3) :    
        ind += 1
        MSE_noise = []

        for eta in noise :
            designMatrix = DesignMatrix('polynomial2D', 5)
            if ind == 0 :
                leastSquares = LeastSquares(backend='manual', method='ols')
            else :
                leastSquares = LeastSquares(backend='manual', method='ridge')
            
            leastSquares.setLambda(lambda_)
            bootstrap    = Bootstrap(leastSquares, designMatrix)

            N = int(1000)
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

            # Different noise, test data
            N = int(1000)
            x = np.random.rand(N)
            y = np.random.rand(N)
            x_data = np.zeros(shape=(N,2))
            x_data[:,0] = x
            x_data[:,1] = y
            y_data = np.zeros(shape=(N))
            computeFrankeValues(x_data, y_data)
            y_data_noise = y_data +  eta * np.random.standard_normal(size=N)

            X = designMatrix.getMatrix(x_data)
            leastSquares.X = X
            leastSquares.predict()
            leastSquares.y = y_data_noise

            MSE.append(leastSquares.MSE())
            R2. append(leastSquares.R2())

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        if ind == 0 :
            ax1.loglog(noise, np.array(MSE_noise), colors[ind]+'--', markersize=1, label=r"OLS")
        else :
            ax1.loglog(noise, np.array(MSE_noise), colors[ind]+'-', markersize=1, label=r"$\lambda=10^{%d}$"%(int(np.log10(lambda_))))
        plt.ylabel(r"MSE", fontsize=10)
        plt.xlabel(r"noise scale $\eta$", fontsize=10)
        plt.subplots_adjust(left=0.2,bottom=0.2)

        #ax1.set_ylim([0.95*min(min(MSE_noise), min(R2_noise)), 1.05*(max(max(MSE_noise), max(R2_noise)))])
        
    ax1.legend()
    #plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'MSE_ridge_noise.png'), transparent=True, bbox_inches='tight')
    plt.show()


def plot_beta_ridge() :
    beta = []
    betaVariance = []
    MSE = []
    R2 = []

    k = 10000
    fig, ax1 = plt.subplots()
    plt.rc('text', usetex=True)


    @jit(nopython=True, cache=True)
    def computeFrankeValues(x_data, y) :    
        N = x_data.shape[0]
        for i in range(N) :
            y[i] = franke(x_data[i,0], x_data[i,1])



    ind = -1
    lam = np.logspace(-3, 5, 20)

    for lambda_ in lam :
        if ind == 0 :
            leastSquares = LeastSquares(backend='manual', method='ols')
        else : 
            leastSquares = LeastSquares(backend='manual', method='ridge')

        designMatrix = DesignMatrix('polynomial2D', 3)
        bootstrap    = Bootstrap(leastSquares, designMatrix)
        leastSquares.setLambda(lambda_)
        ind += 1
        
        N = int(1e4)
        x = np.random.rand(N)
        y = np.random.rand(N)
        x_data = np.zeros(shape=(N,2))
        x_data[:,0] = x
        x_data[:,1] = y
        y_data = np.zeros(shape=(N))
        computeFrankeValues(x_data, y_data)
        eta = 1.0
        y_data_noise = y_data +  eta * np.random.standard_normal(size=N)

        bootstrap.resample(x_data, y_data_noise, k)

        

        MSE.         append(leastSquares.MSE())
        R2.          append(leastSquares.R2())
        beta.        append(bootstrap.beta)
        betaVariance.append(bootstrap.betaVariance)

        leastSquares.y = y_data
        MSE.append(leastSquares.MSE())
        R2. append(leastSquares.R2())

    beta = np.array(beta)
    betaVariance = np.array(betaVariance)

    monomial = ['1',
                'x',
                'y',
                'x^2',
                'xy',
                'y^2',
                'x^3',
                'x^2y',
                'xy^2',
                'y^3']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

    for i in range(10) :
        plt.errorbar(lam[1:], beta[1:,i], 
                        yerr=2*betaVariance[1:,i], 
                        fmt='-o',
                        markersize=2,
                        linewidth=1,
                        color=colors[i],
                        elinewidth=0.5,
                        capsize=2,
                        capthick=0.5,
                        label=r"$\beta_{%s}$"%(monomial[i]))
    plt.rc('text', usetex=True)
    plt.ylabel(r"$\beta_j$",  fontsize=10)
    plt.xlabel(r"shrinkage parameter $\lambda$",  fontsize=10)
    plt.subplots_adjust(left=0.2,bottom=0.2)
    plt.legend(fontsize=8)

    for i in range(10) :
        plt.errorbar(1e-3, beta[0,i], 
                            yerr=2*betaVariance[0,i], 
                            fmt='-o',
                            markersize=2,
                            linewidth=1,
                            color=colors[i],
                            elinewidth=0.5,
                            capsize=2,
                            capthick=0.5)

    fig.gca().set_xscale('log')
    #plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'beta_ridge.png'), transparent=True, bbox_inches='tight')

    plt.show()

    

if __name__ == '__main__':
    part_b()
    #plot_beta_ridge()
    



