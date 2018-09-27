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

def part_a(plotting=False) :
    MSE_degree          = []
    R2_degree           = []
    betaVariance_degree = []

    for degree in [2,3,4,5]: #,6,7,8,9] :
        designMatrix = DesignMatrix('polynomial2D', degree)
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
        bootstrap.resample(x_data, y_data, 1000)
        
        MSE_degree.         append(leastSquares.MSE())
        R2_degree.          append(leastSquares.R2())
        betaVariance_degree.append(bootstrap.betaVariance)
        if plotting :
            print("MSE: ", MSE_degree[-1])
            print("R2:  ", R2_degree[-1])
            print("Beta Variance: ")
            for b in betaVariance_degree[-1] : print(b)
            print("Beta: ")
            for b in leastSquares.beta : print(b)
            print(" ")

            M = 100
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            x = np.linspace(0, 1, M)
            y = np.linspace(0, 1, M)
            X, Y = np.meshgrid(x,y)
            x_data = np.vstack([X.ravel(), Y.ravel()]).T
            
            # When plotting the Franke function itself, we use these lines.
            yy_data = np.zeros(shape=(x_data.data.shape[0]))
            computeFrankeValues(x_data, yy_data)

            # When plotting the linear regression model:
            XX = designMatrix.getMatrix(x_data)
            leastSquares.X = XX
            y_data = leastSquares.predict()

            Z = np.reshape(y_data.T, X.shape)
            ZF = np.reshape(yy_data.T, X.shape)

            #ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,linewidth=0, antialiased=False)
            ax.plot_surface(X,Y,abs(Z-ZF),cmap=cm.coolwarm,linewidth=0, antialiased=False)
            ax.set_zlim(-0.10, 1.40)
            ax.zaxis.set_major_locator(LinearLocator(5))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            ax.view_init(30, 45)

            #plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'franke.png'), transparent=True, bbox_inches='tight')
            #plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'OLS'+str(degree)+'.png'), transparent=True, bbox_inches='tight')
            #plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'OLS'+str(degree)+'_diff.png'), transparent=True, bbox_inches='tight')
            plt.show()

            print("\nMSE :")
            print(MSE_degree)
            print("\nR2 :")
            print(R2_degree)
            print("\nσ²(β) :")
            print(betaVariance_degree)

    return MSE_degree, R2_degree, betaVariance_degree

def plot_betaVariance() :
    MSE, R2, var = part_a()
    fig = plt.figure()
    ax = fig.gca()

    colors = [  
                'r-o',
                'b-o',
                'g-o',
                'k-o',
                'y-o',
                'c-o'
            ]
    var = [None, None, *var]
    for i in var : 
        print(i)
    print(" ")


    plot_var = [None for i in range(6)]
    for k in range(2,5+1) : # polynomial degree
        plot_var[k]   = [None for i in range(k+1)]
        
        plot_var[k][0] = var[k][0]
        plot_var[k][1] = np.mean(var[k][1:3])
        
        for d in range(2,k+1) : # monomials in this degree
            n  = int(round(d*(d+3)/2))
            nn = int(round((d-1)*((d-1)+3)/2))
            N = n-nn
            #print(n,nn,N)

            plot_var[k][d] = np.mean(var[k][nn+1:n+1])
            #print(var[k][nn+1:n+1])
    """
    """
    print(" ")
    plot_var = plot_var[2:]
    print(plot_var)
    print(" ")
    for i in plot_var : print(i)
    print(" ")
    print(plot_var[:][1])
    
    b0 = [plot_var[0][0], plot_var[1][0], plot_var[2][0], plot_var[3][0]]
    b1 = [plot_var[0][1], plot_var[1][1], plot_var[2][1], plot_var[3][1]]
    b2 = [plot_var[0][2], plot_var[1][2], plot_var[2][2], plot_var[3][2]]
    b3 = [                plot_var[1][3], plot_var[2][3], plot_var[3][3]]
    b4 = [                                plot_var[2][4], plot_var[3][4]]
    b5 = [                                                plot_var[3][5]]

    plt.semilogy(range(2,6), b0, colors[0], markersize=3)
    plt.semilogy(range(2,6), b1, colors[1], markersize=3)
    plt.semilogy(range(2,6), b2, colors[2], markersize=3)
    plt.semilogy(range(3,6), b3, colors[3], markersize=3)
    plt.semilogy(range(4,6), b4, colors[4], markersize=3)
    plt.semilogy(range(5,6), b5, colors[5], markersize=3)
    
    plt.rc('text', usetex=True)
    #plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #plt.rc('font',**{'family':'serif','serif':['Palatino']})    plt.xlabel(r"$p$", fontsize=16)
    plt.xlabel(r"$p$", fontsize=10)
    plt.ylabel(r"mean $ \sigma^2(\beta_{p})$", fontsize=10)
    plt.legend([r"intercept", r"$\beta_0$", r"$\beta_1$", r"$\beta_2$", r"$\beta_3$", r"$\beta_4$", r"$\beta_5$"], fontsize=10)

    plt.subplots_adjust(left=0.2,bottom=0.2)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'beta_variance_OLS.png'), transparent=True, bbox_inches='tight')
    plt.show()


    #ax.plot()



def plot_terrain(file_number=1) :
    fileName = os.path.join(os.path.dirname(__file__),  
                            'data', 
                            'SRTM_data_Norway_' + str(file_number) + '.tif')
    image = Image.open(fileName, mode='r')
    print(image.mode)
    print(image.size)
    image.mode = 'I'
    #image.show()
    x = np.linspace(0, 1, image.size[0])
    y = np.linspace(0, 1, image.size[1])
    X,Y = np.meshgrid(x,y)
    Z = np.array(image)
    Z = Z - np.min(Z)
    Z = Z / np.max(Z)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(30, 45+90)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'terrain'+str(file_number)+'.png'), transparent=True, bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    #part_a(plotting=True)
    plot_betaVariance()
    #plot_terrain()




