import os
import sys
import numpy as np
import functools
import matplotlib.pyplot as plt

# Add the src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'project1', 'src'))

from ising          import Ising
from leastSquares   import LeastSquares


def setup(L = 40, N = 1000, train = 0.4) :
    ising   = Ising(L,N)
    ols     = LeastSquares(backend='manual', method='ols')
    ridge   = LeastSquares(backend='manual', method='ridge')
    lasso   = LeastSquares(backend='skl',    method='lasso')

    X_total, y_total = ising.generateDesignMatrix1D()

    N_train = int(train * N)
    N_test  = N - N_train

    X_train = X_total[:N_train]
    y_train = y_total[:N_train]

    X_test  = X_total[N_train:N_train+N_test]
    y_test  = y_total[N_train:N_train+N_test]
    
    return ising, ols, ridge, lasso, X_train, X_test, y_train, y_test    

def MSE_R2_as_function_of_lambda(L = 40, N = 1000, train=0.4, plotting=False) :
    _, ols, ridge, lasso, X_train, X_test, y_train, y_test = setup(L,N,train)

    ols.fit(X_train, y_train)
    ols.predict(X_test)
    ols.y = y_test

    # Copy the value for convenience when plotting later.
    MSE_ols = [ols.MSE() for i in range(2)]
    R2_ols  = [ols.R2()  for i in range(2)]

    MSE_ridge = []
    MSE_lasso = []
    R2_ridge  = []
    R2_lasso  = []

    for method in ['ridge','lasso'] :
        if method == 'ridge' :
            fitter = ridge
        elif method == 'lasso' :
            fitter = lasso

        lambdas = np.logspace(-3, 5, 10)
        for lambda_ in lambdas :
            fitter.setLambda(lambda_)
            fitter.fit(X_train, y_train)
            fitter.predict(X_test)
            fitter.y = y_test

            if method == 'ridge' :
                MSE_ridge.append(fitter.MSE())
                R2_ridge .append(fitter.R2())
            elif method == 'lasso' :
                MSE_lasso.append(fitter.MSE())
                R2_lasso .append(fitter.R2())

    MSE_ols   = np.array(MSE_ols)
    MSE_ridge = np.array(MSE_ridge)
    MSE_lasso = np.array(MSE_lasso)

    R2_ols   = np.array(R2_ols)
    R2_ridge = np.array(R2_ridge)
    R2_lasso = np.array(R2_lasso)    

    if plotting :
        plt.rc('text', usetex=True)
        plt.loglog([lambdas[0], lambdas[-1]], MSE_ols,   marker='o', markersize=2, label=r'OLS')
        plt.loglog(lambdas,                   MSE_ridge, marker='o', markersize=2, label=r'Ridge')
        plt.loglog(lambdas,                   MSE_lasso, marker='o', markersize=2, label=r'Lasso')
        plt.legend(fontsize=10)
        plt.xlabel(r'shrinkage parameter $\lambda$', fontsize=10)
        plt.ylabel(r'MSE',                           fontsize=10)
        plt.subplots_adjust(left=0.2,bottom=0.2)
        plt.show()

        plt.figure()
        plt.loglog([lambdas[0], lambdas[-1]], 1.0 - R2_ols,   marker='o', markersize=2, label=r'OLS')
        plt.loglog(lambdas,                   1.0 - R2_ridge, marker='o', markersize=2, label=r'Ridge')
        plt.loglog(lambdas,                   1.0 - R2_lasso, marker='o', markersize=2, label=r'Lasso')
        plt.legend(fontsize=10)
        plt.xlabel(r'shrinkage parameter $\lambda$', fontsize=10)
        plt.ylabel(r'$1-R^2$ score',                 fontsize=10)
        plt.subplots_adjust(left=0.2,bottom=0.2)
        plt.show()



if __name__ == '__main__':
    np.random.seed(2018)

    MSE_R2_as_function_of_lambda(L=40, N=1000, train=0.2, plotting=True)




