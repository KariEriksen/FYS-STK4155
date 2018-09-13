import numpy as np
from sklearn import linear_model


class LeastSquares :
    def __init__(self, backend='normal') :
        self.backend = backend

    def setBackend(self, backend) :
        self.backend = backend

    def fit(self, X, y) :
        if self.backend == 'manual' :
            self._manualFit(X,y)
        elif ((self.backend == 'sklearn') or (self.backend == 'skl')):
            self._sklFit(X,y)
        else :
            raise ValueError('Backend <' + self.backend + '> not recognized.')

        return self.beta

    def _sklFit(self, X,y) :
        self.regression = linear_model.LinearRegression()
        self.regression.fit(X,y)
        self.beta = self.regression.coef_

    def _manualFit(self, X,y) :
        self.beta = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)), np.dot(np.transpose(X),y))
