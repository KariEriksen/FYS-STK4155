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


    def generateStates1D(self, systemSize=None, numberOfStates=None) :
        self.systemSize     = systemSize     if (systemSize is not None)     else self.systemSize
        self.numberOfStates = numberOfStates if (numberOfStates is not None) else self.numberOfStates

        if (self.systemSize is None) or (self.numberOfStates is None) :
            raise ValueError("System size and/or number of states not specified. ")

        states  = np.random.choice([-1, 1], size=(self.numberOfStates, self.systemSize))
        J       = np.zeros((self.systemSize, self.systemSize),)
        np.fill_diagonal(J[:,1:], -1)
        J[-1,0] = -1
        E       = np.einsum('...i,ij,...j->...', states, J, states)
        
        #EE = np.dot(states[0,:], np.dot(J, states[0,:]))

        #EE = states @ (J @ states.T)
        #EE = np.diag(EE)

        #print(abs(E-EE))
        #print(EE)
        return states, E

            



if __name__ == '__main__':
    ising = Ising()
    L = 40
    states, E = ising.generateStates1D(L, 100)
    
    states=np.einsum('...i,...j->...ij', states, states)

    shape=states.shape
    states=states.reshape((shape[0],shape[1]*shape[2]))
    # build final data set
    Data=[states,E]
    #for s in Data[0] : print(s)
    #print(Data[0].shape)
    np.set_printoptions(precision=3,suppress=True)

    ols = LeastSquares(backend='manual', method='ridge')
    ols.setLambda(0.1)
    beta = ols.fit(states, E)
    print(np.reshape(beta, (L,L)))
    print(ols.MSE())
