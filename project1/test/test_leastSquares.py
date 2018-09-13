import os
import sys
import pytest
import numpy as np

# Add the src/ directory to the python path so we can import the code 
# we need to test directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from leastSquares import LeastSquares



def test_franke() :
    """Tests the Least Squares class as implemented in leastSquares.py

    The tests comprise fitting of models to known data.
    """

    # Ensure fitting polynomials of order 1 through 5 to y(x)=x results
    # in the beta corresponding to the x term equal to 1.0 and all other
    # beta values zero.
    for method in ['manual', 'skl'] :
        N = 10
        x = np.linspace(0,1,N)
        y = x

        for i in range(2,5+1) :
            P = i
            X = np.zeros(shape=(N,P))
            X[:,0] = 1.0

            for j in range(1,P) :
                X[:,j] = x**j

            print(method)
            OLS = LeastSquares(backend=method)
            beta = OLS.fit(X,y)
            print(beta)
            
            assert beta[0] == pytest.approx(0.0, abs=1e-10)
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

    print(X)
    OLS = LeastSquares(backend='manual')
    beta_manual = OLS.fit(X,y)

    OLS = LeastSquares(backend='skl')
    beta_skl    = OLS.fit(X,y)

    print(beta_manual)
    print(beta_skl)

    assert beta_manual == pytest.approx(beta_skl, abs=1e-10)



