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
    N = 10
    x = np.linspace(0,1,N)
    y = x

    for i in range(2,5+1) :
        P = i
        X = np.zeros(shape=(N,P))
        X[:,0] = 1.0

        for j in range(1,P) :
            X[:,j] = x**j

        OLS = LeastSquares(backend='manual')
        beta = OLS.fit(X,y)
        
        assert beta[0] == pytest.approx(0.0, abs=1e-10)
        assert beta[1] == pytest.approx(1.0, abs=1e-10)
        for j in range(2,P) :
            assert beta[j] == pytest.approx(0.0, abs=1e-10)

