import os
import sys
import pytest
import numpy as np
import random
import matplotlib.pyplot as plt

# Add the src/ directory to the python path so we can import the code 
# we need to test directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from bootstrap    import Bootstrap
from leastSquares import LeastSquares
from designMatrix import DesignMatrix


def test_bootstrap_resample() :
    """Tests the resample method of the Bootstrap class

    Tests comprise of simple resampling cases for which we can evalue 
    the exact answer by hand.
    """

    # All data is the same, variance should be zero.
    OLS = LeastSquares()
    DM  = DesignMatrix('polynomial', 3)
    bootstrap = Bootstrap(OLS, DM)

    x = np.ones(10)*2.25
    y = x**3

    bootstrap.resample(x, y, 10)
    assert bootstrap.betaVariance == pytest.approx(np.zeros(4), abs=1e-15)

    # Ensure that larger noise in the data set gives larger computed
    # variance in the beta values from resampling.
    functions = {
        0:  lambda x: np.sin(x),
        1:  lambda x: np.cos(x),
        2:  lambda x: np.sin(2*x),
        3:  lambda x: np.cos(2*x),
        4:  lambda x: np.sin(3*x),
        5:  lambda x: np.cos(3*x),
        6:  lambda x: np.sin(4*x),
        7:  lambda x: np.cos(4*x),
        8:  lambda x: np.sin(5*x),
        9:  lambda x: np.cos(5*x),
        10: lambda x: np.sin(x)**2,
        11: lambda x: np.cos(x)**2,
        12: lambda x: np.sin(2*x)**2,
        13: lambda x: np.cos(2*x)**2,
        14: lambda x: np.sin(3*x)**2,
        15: lambda x: np.cos(3*x)**2,
    }
    DM = DesignMatrix(lambda j,x: functions[j](x), 9)
    OLS = LeastSquares()
    bootstrap = Bootstrap(OLS, DM)
    N  = 100
    x  = np.linspace(0, 2*np.pi, N)
    meanBetaVariance = np.zeros(6)

    ind = 0
    for noiseScale in [0.0, 0.1, 1.0] :
        y = np.sin(1.5*x) - 0.5*np.cos(2*x)**2 + np.random.normal(0, noiseScale, N)
        bootstrap.resample(x, y, 100)
        meanBetaVariance[ind] = np.mean(bootstrap.betaVariance)
        if ind > 0 :
            assert meanBetaVariance[ind-1] < meanBetaVariance[ind]
        ind += 1
