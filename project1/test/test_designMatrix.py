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


def test_DesignMatrix_function() :
    """Tests the function method of the DesignMatrix class

    The tests comprise setting up design matrices with different
    functions and comparing to manually setup matrices.
    """
    class f1 :
        def __init__(self, degree) :
            self.degree = degree

        def __call__(self, i, x) :
            if i > self.degree :
                raise ValueError("Specified function index is larger than the number of available functions.")
            if i == 0 :
                return self._f0(x)
            elif i == 1 :
                return self._f1(x)
            elif i == 2 :
                return self._f2(x)
            elif i == 3 :
                return self._f3(x)
            elif i == 4 : 
                return self._f4(x)
            elif i == 5 : 
                return self._f5(x) 
            elif i == 6 :
                return self._f6(x)

        def _f1(self, x) :
            return np.cos(x)
        def _f2(self, x) :
            return np.sin(x)
        def _f3(self, x) :
            return np.tan(x)
        def _f4(self, x) :
            return np.cosh(x)
        def _f5(self, x) :
            return np.sinh(x)
        def _f6(self, x) :
            return np.tanh(x)

    # 1 function
    numberOfFunctions = 1
    f = f1(numberOfFunctions)
    x = np.array([np.pi/2])
    DM = DesignMatrix(f, numberOfFunctions)
    X  = DM.getMatrix(x)
    X_true = np.array([ [1.0,  np.cos(np.pi/2)] ])
    assert X == pytest.approx(X_true, abs=1e-15)

    # 2 functions
    numberOfFunctions = 2
    f = f1(numberOfFunctions)
    x = np.array([np.pi/2, np.pi/3])
    DM = DesignMatrix(f, numberOfFunctions)
    X  = DM.getMatrix(x)
    X_true = np.array([ [1.0,  np.cos(np.pi/2), np.sin(np.pi/2)],
                        [1.0,  np.cos(np.pi/3), np.sin(np.pi/3)] ])
    assert X == pytest.approx(X_true, abs=1e-15)

    # 3 functions
    numberOfFunctions = 3
    f = f1(numberOfFunctions)
    x = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/6])
    DM = DesignMatrix(f, numberOfFunctions)
    X  = DM.getMatrix(x)
    X_true = np.array([ [1.0,  np.cos(np.pi/2), np.sin(np.pi/2), np.tan(np.pi/2)],
                        [1.0,  np.cos(np.pi/3), np.sin(np.pi/3), np.tan(np.pi/3)],
                        [1.0,  np.cos(np.pi/4), np.sin(np.pi/4), np.tan(np.pi/4)],
                        [1.0,  np.cos(np.pi/5), np.sin(np.pi/5), np.tan(np.pi/5)],
                        [1.0,  np.cos(np.pi/6), np.sin(np.pi/6), np.tan(np.pi/6)] ])
    assert X == pytest.approx(X_true, abs=1e-15)

    # 4 functions
    numberOfFunctions = 4
    f = f1(numberOfFunctions)
    x = np.array([np.pi/2, np.pi/3])
    DM = DesignMatrix(f, numberOfFunctions)
    X  = DM.getMatrix(x)
    X_true = np.array([ [1.0,  np.cos(np.pi/2), np.sin(np.pi/2), np.tan(np.pi/2), np.cosh(np.pi/2)],
                        [1.0,  np.cos(np.pi/3), np.sin(np.pi/3), np.tan(np.pi/3), np.cosh(np.pi/3)] ])
    assert X == pytest.approx(X_true, abs=1e-15)

    # 6 functions
    numberOfFunctions = 6
    f = f1(numberOfFunctions)
    x = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/6, np.pi/7, np.pi/8, np.pi/9])
    DM = DesignMatrix(f, numberOfFunctions)
    X  = DM.getMatrix(x)
    X_true = np.array([ [1.0,  np.cos(np.pi/2), np.sin(np.pi/2), np.tan(np.pi/2), np.cosh(np.pi/2), np.sinh(np.pi/2), np.tanh(np.pi/2)],
                        [1.0,  np.cos(np.pi/3), np.sin(np.pi/3), np.tan(np.pi/3), np.cosh(np.pi/3), np.sinh(np.pi/3), np.tanh(np.pi/3)],
                        [1.0,  np.cos(np.pi/4), np.sin(np.pi/4), np.tan(np.pi/4), np.cosh(np.pi/4), np.sinh(np.pi/4), np.tanh(np.pi/4)],
                        [1.0,  np.cos(np.pi/5), np.sin(np.pi/5), np.tan(np.pi/5), np.cosh(np.pi/5), np.sinh(np.pi/5), np.tanh(np.pi/5)],
                        [1.0,  np.cos(np.pi/6), np.sin(np.pi/6), np.tan(np.pi/6), np.cosh(np.pi/6), np.sinh(np.pi/6), np.tanh(np.pi/6)],
                        [1.0,  np.cos(np.pi/7), np.sin(np.pi/7), np.tan(np.pi/7), np.cosh(np.pi/7), np.sinh(np.pi/7), np.tanh(np.pi/7)],
                        [1.0,  np.cos(np.pi/8), np.sin(np.pi/8), np.tan(np.pi/8), np.cosh(np.pi/8), np.sinh(np.pi/8), np.tanh(np.pi/8)],
                        [1.0,  np.cos(np.pi/9), np.sin(np.pi/9), np.tan(np.pi/9), np.cosh(np.pi/9), np.sinh(np.pi/9), np.tanh(np.pi/9)] ])
    assert X == pytest.approx(X_true, abs=1e-15)







