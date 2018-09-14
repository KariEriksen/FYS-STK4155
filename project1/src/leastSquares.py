import numpy as np
from sklearn import linear_model, metrics


class LeastSquares :
    def __init__(self, backend='normal') :
        self.backend        = backend
        self.fitDone        = False

        self.regression     = None
        
        self.X              = None
        self.y              = None
        self.yHat           = None
        self.beta           = None
        
        self._MSE           = None
        self._R2            = None
        self._betaVariance  = None

    def setBackend(self, backend) :
        self.backend = backend
        self.fitDone = False

    def _checkManualBackend(self) :
        if self.backend == 'manual' :
            return True
        elif ((self.backend == 'sklearn') or (self.backend == 'skl')):
            return False
        else :
            raise ValueError('Backend <' + self.backend + '> not recognized.')

    def _checkFitDoneAndManualBackend(self) :
        manual = self._checkManualBackend()
        if not self.fitDone :
            raise Warning("This action requires a fit to be performed prior to being executed")
        return manual

    def fit(self, X, y) :
        self.X = X
        self.y = y

        if self._checkManualBackend() :
            self._manualFit(X,y)
        else :
            self._sklFit(X,y)
        
        self.fitDone = True
        return self.beta

    def _sklFit(self, X, y) :
        self.regression = linear_model.LinearRegression(fit_intercept=True)
        self.regression.fit(X,y)
        self.beta = self.regression.coef_
        self.beta[0] = self.regression.intercept_

    def _manualFit(self, X,y) :
        self.beta = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)), np.dot(np.transpose(X),y))

    def predict(self, X=None) :
        if X != None :
            self.X = X

        if self._checkFitDoneAndManualBackend() :
            self._manualPredict()
        else :
            self._sklPredict()
        return self.yHat

    def _manualPredict(self) :
        self.yHat = np.dot(self.X, self.beta)

    def _sklPredict(self) :
        self.yHat = self.regression.predict(self.X)-self.beta[0]

    def meanSquaredError(self) :
        return self.MSE()

    def MSE(self) :
        if self._checkFitDoneAndManualBackend() :
            self._manualMSE()
        else :
            self._sklMSE()
        return self._MSE

    def _manualMSE(self) :
        if self.yHat == None :
            self._manualPredict()
        N = self.yHat.size
        self._MSE = np.sum((self.y - self.yHat)**2) / N


    def _sklMSE(self) :
        if self.yHat == None :
            self.yHat = self._sklPredict()
        self._MSE = metrics.mean_squared_error(self.y, self.yHat)







