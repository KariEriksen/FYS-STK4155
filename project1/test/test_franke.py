import os
import sys
import pytest
import numpy as np

# Add the src/ directory to the python path so we can import the code 
# we need to test directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from franke import franke


def test_franke() :
    """Tests the Franke function as implemented in franke.py

    The tests comprise evaluation and comparison against a MATLAB implementation
    of the same function [1].

    [1] https://www.sfu.ca/~ssurjano/Code/franke2dm.html
    """

    x = [     0.5185949425105382,
              0.6489914927123561,
              0.6596052529083072,
              0.8003305753524015,
              0.9729745547638625
        ]
    y = [     0.0834698148589140,
              0.1331710076071617,
              0.4323915037834617,
              0.4537977087269195,
              0.8253137954020456
        ]
    f = [     0.4488306370927234,
              0.4063778975108695,
              0.4875600327917881,
              0.4834151909828157,
              0.0479911637101943
        ]
    for i in range(5) :
        assert franke(x[i], y[i]) == pytest.approx(f[i], abs=1e-15)
    