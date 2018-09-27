import numpy as np
from sklearn import linear_model, metrics


class LeastSquares :
    """Perform ordinary least squares fits manually or with scikit-learn
    
    The method(s) used (manual, written from scratch or using scikit-learn) 
    is determined by the backend parameter given to the constructor (or 
    changed later using the setBackend() method). 
    """

    def __init__(self, method='ols', backend='manual') :
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
        ValueError 
            If the backend string is not equal to 'manual', 'skl', or 
            'sklearn'
        """
        self.backend        = backend
        self.method         = method
        self.lambda_        = 0
        self.lambdaSet      = False
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


    def setLambda(self, lambda_) :
        self.lambda_ = lambda_
        self.lambdaSet = True

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
            if self.method == 'ols' :
                self._manualFit(X,y)
            elif self.method == 'ridge' :
                self._manualFitRidge(X,y)
            elif self.method == 'lasso' :
                raise NotImplementedError("Lasso regression is not implemented from scratch. Use backend='skl'")
        else :
            if self.method == 'ols' :
                self._sklFit(X,y)
            elif self.method == 'ridge' :
                self._sklFitRidge(X,y)
            elif self.method == 'lasso' :
                self._sklFitLasso(X,y)
        
        self.fitDone = True
        return self.beta


    def _manualFit(self, X, y) :
        """The manual version of the fitting method

        Performs the ordinary linear regression fit using the *from scratch* 
        implementation. The matrix calculation performed corresponds to 

            b = inv(Xt X) * Xt * y,

        where b is the vector of beta values, Xt denotes the tranpose of X and 
        inv(A) is the inverse of the matrix A.

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
        """
        self.beta = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)), np.dot(np.transpose(X),y))


    def _manualFitRidge(self, X, y) :
        if self.lambdaSet == False :
            raise ValueError("No lambda value set for Ridge regression. Use LeastSquares.setLambda()")
        #U,S,Vt = np.linalg.svd(X, full_matrices=False)
        #self.beta = np.dot(np.dot(np.dot(np.dot(np.transpose(Vt), np.linalg.inv(np.dot(np.transpose(S),S)+self.lambda_*np.eye(S.shape[0]))),S),np.transpose(U)),y)
        self.beta = np.dot(np.linalg.inv(np.dot(np.transpose(X),X) + self.lambda_ * np.eye(X.shape[1])), np.dot(np.transpose(X),y))


    def _sklFitRidge(self, X, y) :
        if self.lambdaSet == False :
            raise ValueError("No lambda value set for Ridge regression. Use LeastSquares.setLambda()")
        self.regression = linear_model.Ridge(fit_intercept=True, alpha=self.lambda_, solver='svd')
        self.regression.fit(X,y)
        self.beta = self.regression.coef_
        self.beta[0] = self.regression.intercept_

    def _sklFitLasso(self, X, y) :
        if self.lambdaSet == False :
            raise ValueError("No lambda value set for Lasso regression. Use LeastSquares.setLambda()")
        self.regression = linear_model.Lasso(fit_intercept=True, alpha=self.lambda_)
        self.regression.fit(X,y)
        self.beta = self.regression.coef_
        self.beta[0] = self.regression.intercept_        

    def _sklFit(self, X, y) :
        """The scikit-learn version of the fitting method

        Performs the ordinary linear regression fit using the scikit-learn 
        functionality. Note that the self.beta array resulting from getting
        linear_model.LinearRegression.coef_ does not contain the intercept
        by default, even when LinearRegression(fit_intercept=True) is used. 
        In order to have self.beta correspond to the same quantity in both
        the 'manual' and 'skl'/'sklearn' cases, we modify it in the scikit-
        learn case to contain the linear_model.LinearRegression.intercept_ 
        in self.beta[0]. 

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
        """
        self.regression = linear_model.LinearRegression(fit_intercept=True)
        self.regression.fit(X,y)
        self.beta = self.regression.coef_
        self.beta[0] = self.regression.intercept_


    def predict(self, X=None) :
        """Calculates the prediction of the linear model based on the input data after having performed a fit

        Performs the computation of the prediction of the optimized linear model
        based on the input data. If the backend is 'manual', the *from scratch* 
        implementation is called. If the backend is 'skl' or 'sklearn', the 
        scikit-learn linear_model.LinearRegression.predict() method is called.

        Returns
        -------
        self.yHat : numpy.array
            The prediction array, a 1D numpy array of prediction values, 
            dimensions (n_dataPoints, 1)

        Raises
        ------
        Warning
            If a fit has not been performed yet, a warning is raised telling the 
            user that getting a prediction out of the model before the fit is 
            done does not make much sense.
        """
        if X is not None :
            self.X
            
        if self._checkFitDoneAndManualBackend() :
            self._manualPredict()
        else :
            self._sklPredict()
        return self.yHat


    def _manualPredict(self) :
        """Manual version of the prediction calculation

        Performs a simple matrix multiplication of the design matrix and the 
        computed beta values resulting from the fit to obtain the model
        prediction.

        Returns
        -------
        self.yHat : numpy.array
            The prediction array, a 1D numpy array of prediction values, 
            dimensions (n_dataPoints, 1)
        """
        self.yHat = np.dot(self.X, self.beta)


    def _sklPredict(self) :
        """Scikit-learn version of the prediction calculation

        Calls the scikit-learn linear_model.LinearRegression.predict() method
        in order to obtain the prediction. In order to get the prediction with 
        intercept, we need to subtract self.beta[0] which contains the value 
        of linear_model.LinearRegression.intercept_ (see self._sklFit()). This 
        is rougly corresponds to doing metric.mean_squared_error(fit_intercept=True),
        which is not a functionality available in scikit-learn currently.

        Returns
        -------
        self.yHat : numpy.array
            The prediction array, a 1D numpy array of prediction values, 
            dimensions (n_dataPoints, 1)
        """
        self.yHat = self.regression.predict(self.X) - self.beta[0]


    def meanSquaredError(self) :
        """Calculate the mean squared error in the model prediction

        Alias of the self.MSE() method in order to allow calls to both 
        LeastSquares.meanSquaredError() and LeastSquares.MSE(). 

        Returns
        -------
        self._MSE
            The mean squared error in the prediction of the model.
        """
        return self.MSE()


    def MSE(self) :
        """Calculate the mean squared error in the model prediction

        If the backend is 'manual', the *from scratch* implementation is 
        called. If the backend is 'skl' or 'sklearn', the scikit-learn 
        metrics.mean_squared_error() method is called. 

        Returns
        -------
        self._MSE
            The mean squared error in the prediction of the model.
        """
        if self._checkFitDoneAndManualBackend() :
            self._manualMSE()
        else :
            self._sklMSE()
        return self._MSE


    def _manualMSE(self) :
        """Manual version of the mean squared error calculation

        If the prediction has not been calculated previously, it is 
        calculated before the MSE is calculated by a simple numpy.sum 
        of the squared difference of the prediction and the true y data, 
        and a divide by the number of data points.

        Returns
        -------
        self._MSE
            The mean squared error in the prediction of the model.
        """
        if self.yHat is None :
            self._manualPredict()
        N = self.yHat.size
        self._MSE = np.sum((self.y - self.yHat)**2) / N


    def _sklMSE(self) :
        """Scikit-learn version of the mean squared error calculation

        If the prediction has not been calculated previously, it is 
        calculated before the MSE is calculated by calling the
        metrics.mean_squared_error() method.

        Returns
        -------
        self._MSE
            The mean squared error in the prediction of the model.
        """
        if self.yHat is None :
            self._sklPredict()
        self._MSE = metrics.mean_squared_error(self.y, self.yHat)


    def R2(self) :
        """Computes the R2 score of the model prediction 

        Computes the R2 score of the model prediction, after a fit has 
        been made. If the backend is 'manual', the *from scratch* 
        implementation is called. If the backend is 'skl' or 'sklearn', the 
        scikit-learn metrics.r2_score() method is called used. If no fit has 
        been performed prior to calling this method, a warning is raised 
        telling the user that this doesnt make any sense (yet). 

        Returns
        -------
        self._R2
            The R2 score of the model prediction

        Raises
        ------
        Warning
            If a fit has not been performed yet, a warning is raised telling the 
            user that getting a prediction out of the model before the fit is 
            done does not make much sense.
        """
        manual = self._checkFitDoneAndManualBackend()
        if manual :
            self._manualR2()
        else :
            self._sklR2()
        return self._R2


    def _manualR2(self) :
        """Manual version of the R2 score calculation

        If the prediction has not been calculated previously, it is 
        calculated before the R2 score is calculated. The calculation consists of 
        dividing the sum of the square difference between the model prediction
        and the true y values, and the square difference between the model 
        prediction and the mean model prediction value. 

        Returns
        -------
        self._MSE
            The mean squared error in the prediction of the model.
        """
        if self.yHat is None :
            self._manualPredict()
        yMean = (1.0 / self.y.size) * np.sum(self.y)
        self._R2 = 1.0 - np.sum((self.yHat - self.y)**2) / np.sum((self.y - yMean)**2)
        return self._R2


    def _sklR2(self) :
        """Scikit-learn version of the R2 score calculation

        If the prediction has not been calculated previously, it is 
        calculated before the R2 score is calculated. The calculation is done by
        calling the metrics.r2_score() method.

        Returns
        -------
        self._MSE
            The mean squared error in the prediction of the model.
        """
        if self.yHat is None :
            self._sklPredict()
        self._R2 = metrics.r2_score(self.y, self.yHat)
        return self._R2






