import numpy as np
from sklearn import linear_model, metrics


class LeastSquares :
    """Perform ordinary least squares fits manually or with scikit-learn
    
    The method(s) used (manual, written from scratch or using scikit-learn) 
    is determined by the backend parameter given to the constructor (or 
    changed later using the setBackend() method). 
    """

    def __init__(self, backend='normal') :
        """Constructor

        Determines the method(s) used in the fitting and evaluation with
        the provided backend. 

        Parameters
        ----------
        backend : string
            Specifies the methods used, 'manual' for the *from scratch* 
            implementation, or 'sklearn' for the scikit-learn implementations.
            'skl' is also accepted as a synonym of 'sklearn'. Any other string
            input results in a ValueError being raised.

        Raises
        ------
        ValueError : If the backend string is not equal to 'manual', 'skl', or 'sklearn'
        """
        self.backend        = backend
        self._checkManualBackend()

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
        """Change the backend in use

        Change from/to using manual methods or scikit-learn methods after having 
        initialized an instance of LeastSquares.

        Parameters
        ----------
        backend : string
            The new backend to use. Any other string besides 'manual', 'skl', or 
            'sklearn' results in a ValueError being raised.

        Raises
        ------
        ValueError
            If the backend string is not equal to 'manual', 'skl', or 'sklearn'
        """
        self.backend = backend
        self._checkManualBackend()
        self.fitDone = False


    def _checkManualBackend(self) :
        """Checks if the backend is 'manual' or 'skl' / 'sklearn'

        If the backend is 'manual', True is returned. If the backend is 'skl' or 
        'sklearn', False is returned. Any other string found in self.backend results
        in a ValueError.

        Returns
        -------
        bool
            True if self.backend is 'manual', False if self.backend is 'skl' or 
            'sklearn'.

        Raises
        ------
        ValueError
            If the backend string is not equal to 'manual', 'skl', or 'sklearn'
        """        

        if self.backend == 'manual' :
            return True
        elif ((self.backend == 'sklearn') or (self.backend == 'skl')):
            return False
        else :
            raise ValueError('Backend <' + self.backend + '> not recognized.')


    def _checkFitDoneAndManualBackend(self) :
        """Checks if a least squares fit has been performed by this class instance and if the backend is 'manual'

        Returns True if the fit has been performed and self.backend is 'manual', False if the 
        fit has been performed but the backend is 'skl' or 'sklearn'. If no fit has been 
        performed yet, a Warning is raised.

        Returns
        -------
        bool
            True if self.backend is 'manual', False if self.backend is 'skl' or 
            'sklearn'.

        Raises
        ------
        Warning
            If no fit has been performed, a warning is raised to let the user know that
            the action they are trying to perform (e.g. MSE calculation) is not going to 
            be meaningful.
        """
        manual = self._checkManualBackend()
        if not self.fitDone :
            raise Warning("This action requires a fit to be performed prior to being executed")
        return manual


    def fit(self, X, y) :
        """Performs the ordinary least squares fit of the provided data

        Depending on the contents of self.backend, the fit is done using either the 
        *from scratch* implementation or the scikit-learn functionality.

        Parameters
        ----------
        X : numpy.array
            The design matrix, a 2D numpy array, dimensions (n_dataPoints, n_parameters)
        y : numpy.array
            The true data y values, a 1D numpy array, dimension(n_dataPoints, 1)
        
        Returns
        -------
        beta : numpy.array
            The optimized beta parameters from the performed fit

        Raises
        ------
        ValueError
            If the self.backend string is not equal to 'manual', 'skl', or 'sklearn'
        """
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
            self._sklPredict()
        self._MSE = metrics.mean_squared_error(self.y, self.yHat)







