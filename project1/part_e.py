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


def mesh_test() :
    x = np.array([1,2,3,4])
    y = np.array([11,22,33,44])


    print("x", x)
    print("y", y)
    print(" ")

    X,Y = np.meshgrid(x,y)
    print("X",X)
    print("Y",Y)
    print(" ")

    Z = 2*X**2+Y
    print(Z)
    print("===---===")
    X = X[::2,::2]
    Y = Y[::2,::2]
    Z = Z[::2,::2]
    print("X",X)
    print("Y",Y)
    print("Z",Z)



    """
    x_data = np.vstack([X.ravel(), Y.ravel()]).T
    print("x_data[:,0]", x_data[:,0])
    print("x_data[:,1]", x_data[:,1])
    xx = x_data[:,0]
    yy = x_data[:,1]

    ZZ = np.reshape(Z,(-1,1))
    ZZ = ZZ[::1]
    
    xx = xx[::2]
    yy = yy[::]
    print("===")
    print(xx)
    print(yy)
    
    print("x_data stripped")
    print(x_data)
    print("ZZ", ZZ)

    s = int(np.sqrt(x_data.shape[0]))

    XX = np.resize(x_data[:,0], (s,s))
    YY = np.resize(x_data[:,1], (s,s))



    print(" ")
    print("XX", XX)

    print(" ")
    print("YY", YY)

    ZZZ = np.reshape(ZZ, XX.shape)
    print(" ")
    print("ZZZ", ZZZ)
    """


def real_data(file_number=2, plotting=False) :
    fileName = os.path.join(os.path.dirname(__file__),  
                            'data', 
                            'SRTM_data_Norway_' + str(file_number) + '.tif')
    image = Image.open(fileName, mode='r')
    image.mode = 'I'
    x = np.linspace(0, 1, image.size[0])
    y = np.linspace(0, 1, image.size[1])
    X,Y = np.meshgrid(x,y)
    Z = np.array(image)
    Z = Z - np.min(Z)
    Z = Z / np.max(Z)

    print(X.shape)
    print(Y.shape)
    print(Z.shape)

    skip = 30

    XX = X[:-10:2*skip, :-10:skip]
    YY = Y[:-10:2*skip, :-10:skip]
    ZZ = Z[:-10:2*skip, :-10:skip]

    x_train = np.vstack([XX.ravel(), YY.ravel()]).T
    y_train = np.reshape(ZZ, x_train.shape[0])

    skip = 30
    skipstart = 14
    print(XX.shape)

    XX = X[skipstart::2*skip, skipstart::skip]
    YY = Y[skipstart::2*skip, skipstart::skip]
    ZZ = Z[skipstart::2*skip, skipstart::skip]

    x_test = np.vstack([XX.ravel(), YY.ravel()]).T
    y_test = np.reshape(ZZ, x_test.shape[0])

    print(XX.shape)

    print(x_test.shape)
    print(x_train.shape)

    if plotting :
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,linewidth=0, antialiased=False)
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(5))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.view_init(30, 45+90)
        plt.show()

    return x_train, y_train, x_test, y_test


def part_e(plotting=False) :
    x_train, y_train, x_test, y_test = real_data(file_number=2, plotting=False)

    for method in ['ols', 'ridge', 'lasso'] :
        designMatrix = DesignMatrix('polynomial2D', 10)
        leastSquares = LeastSquares(backend='manual', method=method)
        leastSquares.setLambda(1e-3)
        if method == 'lasso' :
            leastSquares.setLambda(1e-4)
        
        X = designMatrix.getMatrix(x_train)
        leastSquares.fit(X,y_train)

        X_test = designMatrix.getMatrix(x_test)
        leastSquares.predict()
        leastSquares.y = y_test
        print(leastSquares.MSE())

        if plotting :
            x = np.linspace(0, 1, 60)
            y = np.copy(x)
            XX,YY = np.meshgrid(x,y)
            ZZ = np.reshape(leastSquares.yHat, XX.shape)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(XX,YY,ZZ, cmap=cm.coolwarm,linewidth=0, antialiased=False)
            ax.set_zlim(-0.10, 1.40)
            ax.zaxis.set_major_locator(LinearLocator(5))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            ax.view_init(30, 45+90)
            plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', method+'terrain.png'), transparent=True, bbox_inches='tight')
            plt.show()
    if plotting :
        x = np.linspace(0, 1, 60)
        y = np.copy(x)
        XX,YY = np.meshgrid(x,y)
        ZZ = np.reshape(y_test, XX.shape)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(XX,YY,ZZ, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(5))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.view_init(30, 45+90)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'test_terrain.png'), transparent=True, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    #real_data()
    #mesh_test()
    part_e(plotting=True)

