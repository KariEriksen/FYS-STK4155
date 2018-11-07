import numpy as np
import scipy 
import warnings
import os 
import sys 
import collections
import matplotlib.pyplot as plt

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

        self.states  = np.random.choice([-1.0, 1.0], size=(self.numberOfStates, self.systemSize))
        self.J  = np.zeros((self.systemSize, self.systemSize),)
        np.fill_diagonal(self.J[:,1:], -1.0)
        self.J[-1,0] = -1.0

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


    def generateTrainingData1D(self, systemSize=None, numberOfStates=None) :
        """
        Performs even sampling, resulting in equal probability for states of all
        possible energies.
        """
        self.systemSize     = systemSize     if (systemSize is not None)     else self.systemSize
        self.numberOfStates = numberOfStates if (numberOfStates is not None) else self.numberOfStates

        L = self.systemSize
        N = self.numberOfStates
        possible_E = len(np.arange(-L, L+1, 4))
        n = N // possible_E
        self.states = np.zeros(shape=(N, L))
        
        self.J  = np.zeros((self.systemSize, self.systemSize),)
        np.fill_diagonal(self.J[:,1:], -1.0)
        self.J[-1,0] = -1.0

        s = np.ones(shape=(n,L))
        s[:,::2] *= -1
        ss = s[0,:]
        s = np.multiply(s, np.expand_dims(np.random.choice([-1,1], size=n),axis=1))
        self.states[:n] = s

        for i in range(1,possible_E-1) :
            s         = np.ones(shape=(n,L))
            s[:,::2] *= -1
            s         = np.multiply(s, np.expand_dims(np.random.choice([-1,1], size=n),axis=1))
            for j in range(n) :
                s[j,np.random.choice(np.arange(0,L,2), replace=False, size=(1,i))] *= -1
            self.states[i*n:(i+1)*n] = s

        s = np.ones(shape=(n,L))
        s = np.multiply(s, np.expand_dims(np.random.choice([-1,1], size=n),axis=1))
        self.states[(possible_E-1)*n:(possible_E-1)*n+s.shape[0]] = s

        if N % possible_E != 0 :
            n = N % possible_E
            self.states[N-n:] = np.random.choice([-1.0, 1.0], size=(n, L))
        np.random.shuffle(self.states)
        
        self.computeEnergy1D()
        return self.states, self.E


            












