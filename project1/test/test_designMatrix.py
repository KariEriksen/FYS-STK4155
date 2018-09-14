import os
import sys
import pytest
import numpy as np
import random
import matplotlib.pyplot as plt

# Add the src/ directory to the python path so we can import the code 
# we need to test directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from designMatrix import DesignMatrix



def test_DesignMatrix_polynomial() :
    """Tests the polynomial method of the DesignMatrix class

    The tests comprise setting up design matrices of different 
    polynomials orders and comparing to manually setup matrices.
    """
    x = np.array([1.0, 2.0, 3.0])
    DM = DesignMatrix('polynomial', 3)
    X  = DM.getMatrix(x)
    X_true = np.array([ [1.0,  1.0,  1.0,  1.0],
                        [1.0,  2.0,  4.0,  8.0],
                        [1.0,  3.0,  9.0, 27.0] ])
    assert X == pytest.approx(X_true, abs=1e-15)


if __name__ == '__main__':
    test_DesignMatrix_polynomial()