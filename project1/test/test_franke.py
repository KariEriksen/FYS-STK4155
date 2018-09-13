import os
import sys
import numpy as np

# Add the src/ directory to the python path so we can import the code 
# we need to test directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_franke() :
	from franke import franke
	x = 1
	y = 2
	assert franke(x,y) == 1

