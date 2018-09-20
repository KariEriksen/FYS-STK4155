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
    DM  = DesignMatrix('polynomial', 2)
    bootstrap = Bootstrap(OLS, DM)

    x = np.ones(10)*2.25
    y = x**3

    bootstrap.resample(x, y, 10)
    assert bootstrap.betaVariance == pytest.approx(np.zeros(3), abs=1e-15)




