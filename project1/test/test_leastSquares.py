import os
import sys
import pytest
import numpy as np
import random
import matplotlib.pyplot as plt

# Add the src/ directory to the python path so we can import the code 
# we need to test directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from leastSquares import LeastSquares



def test_LeastSquares_fit() :
    """Tests the fit method of the Least Squares class

    The tests comprise fitting of models to known data.
    """

    # Ensure fitting polynomials of order 1 through 5 to y(x) = x results
    # in the beta corresponding to the x term equal to 1.0 and all other
    # beta values zero.
    #
    # Secondly we test on y(x)=x + 2 to also make sure the intercept is 
    # calculated correctly.
    for intercept in [0, 2] :
        for method in ['manual', 'skl'] :
            N = 10
            x = np.linspace(0,1,N)
            y = x + intercept

            for i in range(2,5+1) :
                P = i
                X = np.zeros(shape=(N,P))
                X[:,0] = 1.0

                for j in range(1,P) :
                    X[:,j] = x**j

                OLS = LeastSquares(backend=method)
                beta = OLS.fit(X,y)
                
                assert beta[0] == pytest.approx(intercept, abs=1e-10)
                assert beta[1] == pytest.approx(1.0, abs=1e-10)
                for j in range(2,P) :
                    assert beta[j] == pytest.approx(0.0, abs=1e-10)



    # Ensure the backend='manual' and the backend='skl' versions of 
    # LeastSquares.fit(X,y) give the same result.
    N = 5
    P = 5
    x = np.linspace(0,1,N)
    y = x + x**2 - (1.0 - x)**5
    
    X = np.zeros(shape=(N,P))
    X[:,0] = 1.0
    for j in range(1,P) :
        X[:,j] = x**j

    OLS = LeastSquares(backend='manual')
    beta_manual = OLS.fit(X,y)

    OLS = LeastSquares(backend='skl')
    beta_skl    = OLS.fit(X,y)

    assert beta_manual == pytest.approx(beta_skl, abs=1e-10)

def test_LeastSquares_predict() :
    """Tests the predict method of the Least Squares class

    The test is done with a known beta array, comparing results to a known
    MSE value.
    """
    N = 5
    P = 3
    x = np.linspace(0,1,N)
    random.seed(10)
    y = 3*x**2 - 9*x - 2.4*x**5 - 3.1
    X = np.zeros(shape=(N,P))
    X[:,0] = 1.0

    for j in range(1,P) :
        X[:,j] = x**j

    OLS = LeastSquares(backend='skl')
    beta_skl    = OLS.fit(X,y)
    predict_skl = OLS.predict()

    OLS = LeastSquares(backend='manual')
    beta_manual    = OLS.fit(X,y)

    # Ensure the exact same beta value are used by both backend versions.
    OLS.beta       = beta_skl
    predict_manual = OLS.predict()

    assert (predict_manual == pytest.approx(predict_skl, abs=1e-15))


def test_LeastSquares_meanSquaredError() :
    """Tests the meanSquaredError method of the Least Squares class

    The test is done with a known beta array, comparing results to a known
    MSE value.
    
    N = 5
    P = 3
    x = np.linspace(0,1,N)
    random.seed(10)
    y = 3*x**2 - 9*x - 2.4*x**5 - 3.1
    X = np.zeros(shape=(N,P))
    X[:,0] = 1.0

    for j in range(1,P) :
        X[:,j] = x**j

    OLS = LeastSquares(backend='skl')
    beta_skl    = OLS.fit(X,y)
    MSE_skl     = OLS.meanSquaredError()

    OLS = LeastSquares(backend='manual')
    beta_manual = OLS.fit(X,y)

    # Ensure the manual and the skl fit both use the exact same beta 
    # values.
    OLS.beta    = beta_skl
    MSE_manual  = OLS.meanSquaredError()

    assert MSE_skl == pytest.approx(MSE_manual, abs=1e-10)
    
    """





