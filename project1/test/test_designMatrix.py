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
    # Degree 1 polynomial
    x = np.array([2.0])
    DM = DesignMatrix('polynomial', 1)
    X  = DM.getMatrix(x)
    X_true = np.array([ [1.0,  2.0] ])
    assert X == pytest.approx(X_true, abs=1e-15)

    # Degree 2 polynomial
    x = np.array([2.0, 3.0])
    DM = DesignMatrix('polynomial', 2)
    X  = DM.getMatrix(x)
    X_true = np.array([ [1.0,  2.0,  4.0],
                        [1.0,  3.0,  9.0] ])
    assert X == pytest.approx(X_true, abs=1e-15)

    # Degree 3 polynomial
    x = np.array([2.0, 3.0, 4.0])
    DM = DesignMatrix('polynomial', 3)
    X  = DM.getMatrix(x)
    X_true = np.array([ [1.0,  2.0,  4.0,  8.0],
                        [1.0,  3.0,  9.0, 27.0],
                        [1.0,  4.0, 16.0, 64.0] ])
    assert X == pytest.approx(X_true, abs=1e-15)

    # Degree 4 polynomial
    x = np.array([2.0, 3.0, 4.0, 5.0])
    DM = DesignMatrix('polynomial', 4)
    X  = DM.getMatrix(x)
    X_true = np.array([ [1.0,   2.0,   4.0,   8.0,  16.0],
                        [1.0,   3.0,   9.0,  27.0,  81.0],
                        [1.0,   4.0,  16.0,  64.0, 256.0],
                        [1.0,   5.0,  25.0, 125.0, 625.0] ])
    assert X == pytest.approx(X_true, abs=1e-15)

    # Degree 5 polynomial
    x = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    DM = DesignMatrix('polynomial', 5)
    X  = DM.getMatrix(x)
    X_true = np.array([ [1.0,    2.0,    4.0,    8.0,   16.0,    32.0],
                        [1.0,    3.0,    9.0,   27.0,   81.0,   243.0],
                        [1.0,    4.0,   16.0,   64.0,  256.0,  1024.0],
                        [1.0,    5.0,   25.0,  125.0,  625.0,  3125.0],
                        [1.0,    6.0,   36.0,  216.0, 1296.0,  7776.0] ])
    assert X == pytest.approx(X_true, abs=1e-15)

    # Degree 6 polynomial
    x = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    DM = DesignMatrix('polynomial', 6)
    X  = DM.getMatrix(x)
    X_true = np.array([ [1.0,    2.0,    4.0,    8.0,   16.0,     32.0,      64.0],
                        [1.0,    3.0,    9.0,   27.0,   81.0,    243.0,     729.0],
                        [1.0,    4.0,   16.0,   64.0,  256.0,   1024.0,    4096.0],
                        [1.0,    5.0,   25.0,  125.0,  625.0,   3125.0,   15625.0],
                        [1.0,    6.0,   36.0,  216.0, 1296.0,   7776.0,   46656.0],
                        [1.0,    7.0,   49.0,  343.0, 2401.0,  16807.0,  117649.0] ])
    assert X == pytest.approx(X_true, abs=1e-15)

