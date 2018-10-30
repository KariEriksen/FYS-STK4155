import numpy as np
import scipy 
import warnings
import os 
import sys 
import pytest
import copy

# Add the project2/src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuralNetwork  import NeuralNetwork
from activation     import Activation
from costFunction 	import CostFunction

def test_costFunction() :
	pass