import numpy as np
import scipy 
import warnings
import os 
import sys 

# Add the project1/src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'project1', 'src'))

from leastSquares import LeastSquares

warnings.filterwarnings('ignore')


class Ising :
    def __init__(self, systemSize=None, numberOfStates=None) :
        self.systemSize     = systemSize
        self.numberOfStates = numberOfStates
        self.states         = None


    def generateStates1D(self, systemSize=None, numberOfStates=None) :
        self.systemSize     = systemSize     if (systemSize is not None)     else self.systemSize
        self.numberOfStates = numberOfStates if (numberOfStates is not None) else self.numberOfStates

        if (self.systemSize is None) or (self.numberOfStates is None) :
            raise ValueError("System size and/or number of states not specified.")

        self.states  = np.random.choice([-1, 1], size=(self.numberOfStates, self.systemSize))
        self.J  = np.zeros((self.systemSize, self.systemSize),)
        np.fill_diagonal(self.J[:,1:], -1)
        self.J[-1,0] = -1

        return self.states, self.computeEnergy1D()


    def computeEnergy1D(self) :
        self.E = np.einsum('...i,ij,...j->...', self.states, self.J, self.states)
        return self.E

            
    def generateDesignMatrix1D(self, systemSize=None, numberOfStates=None) :
        self.systemSize     = systemSize     if (systemSize is not None)     else self.systemSize
        self.numberOfStates = numberOfStates if (numberOfStates is not None) else self.numberOfStates

        if self.states is None :
            self.generateStates1D()

        self.X = np.einsum('...i,...j->...ij', self.states, self.states)
        self.X = np.reshape(self.X, (self.numberOfStates, self.systemSize**2))
        self.y = self.E

        return self.X, self.y