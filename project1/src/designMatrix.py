import numpy as np


class DesignMatrix :
    """Automate setting up of design matrices for linear regression problems
    """

    def __init__(self, model=None, degree=None) :
        """Constructor of the DesignMatrix class

        Takes as input the type of the desired design matrix and performs
        some checks on the input to make sure it makes sense. The model 
        parameter can either be a callable object (such as a function) or
        a string denoting the type of model requested. If the model argument
        is neither, a TypeError is raised. 

        If the model argument is a callable object *and* the degree 
        argument is also provided, a Warning is raised telling the user 
        that the degree argument will be ignored.

        If the model argument is provided in string form, but the string
        is not 'polynomial', a ValueError is raised. If 'polynomial' is 
        provided as the model argument but the degree argument is not 
        present, a ValueError is also raised. If the degree argument is 
        present but not an integer, a TypeError is raised.

        Parameters
        ----------
        model 
            Either a callable object or a string. In the latter case, only
            'polynomial' is accepted.
        degree : int
            The degree of the polynomial model used, only considered in the
            case of model being the string 'polynomial'.

        Raises
        ------
        Warning
            If the degree argument is provided alongside the model argument
            'polynomial', a warning is raised telling the user that degree
            will be ignored.
        TypeError
            If the model parameter is neither callable nor a string, or if 
            degree is not an int
        ValueError
            If a string different from 'polynomial' is provided for the 
            model argument, or if 'polynomial' is given but no degree 
            parameter is specified
        """
        self.degree = degree
        self.model  = model
        self.type_  = None

        if callable(model) :
            self.type_ = 'function'
            if degree != None :
                raise Warning("When providing a callable object for the DesignMatrix, the degree parameter is ignored.")
        else :
            if isinstance(model, str) :
                if model == 'polynomial' :
                    self.type_ = 'polynomial'
                    if degree != None :
                        if type(degree) == int :
                            self.degree = degree
                        else :
                            raise TypeError("The provided polynomial degree is not an integer.")
                    else :
                        raise ValueError("Using the 'polynomial' type DesignMatrix requires providing a polynomial degree.")
                else :
                    raise ValueError("Model string <" + model +"> not recognized.")
            else :
                raise TypeError("The input model to DesignMatrix is neither a callable object nor a string.")


    def getMatrix(self, x) :
        if self.type_ == 'polynomial' :
            self._getMatrixPolynomial(x)
        elif self._type_ == 'function' :
            raise NotImplementedError("...")
        return self.matrix


    def _getMatrixPolynomial(self, x) :
        N = x.size
        P = self.degree
        self.matrix = np.zeros(shape=(N,P+1))
        self.matrix[:,0] = 1.0
        for j in range(1,P+1) :
            self.matrix[:,j] = x**j













