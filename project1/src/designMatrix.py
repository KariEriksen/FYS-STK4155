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
            The degree of the polynomial model used, or the number of functions
            used in the case of a callable model input.

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
            if degree == None :
                raise ValueError("The degree parameter must specify the number of different functions when initializing DesignMatrix with a callable object.")
            if type(degree) != int :
                raise ValueError("The specified number of different functions must be an integer.")
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
        """Compute the design matrix of the chosen type

        Computes and returns the design matrix of either polynomial or
        custom function type. If a callable object was given in the 
        constructor, the custom function design matrix is setup. If the 
        'polynomial' keyword was given, a polynomial design matrix of the
        specified order is setup.

        In the former case, the self._getMatrixFunction method is called,
        while the latter results in a self._getMatrixPolynomial call.

        Paramters
        ---------
        x : numpy.array
            The data set, a 1D numpy array, used for the construction of
            the design matrix

        Returns
        -------
        self.matrix : numpy.array
            The design matrix, a 2D numpy array
        """
        if self.type_ == 'polynomial' :
            self._getMatrixPolynomial(x)
        elif self.type_ == 'function' :
            self._getMatrixFunction(x)
        return self.matrix


    def _getMatrixPolynomial(self, x) :
        """Computes the design matrix for a polynomial of a given degree

        Computes the design matrix of polynomial type with degree specified 
        in the input to the constructor. The first column contains unity, 
        the subsequent columns are evaluated as 

            ╭     ╮      j
            │  X  │ =  x  
            ╰   ij╯     i
        
        with X being the design matrix and x being the input data set.
        
        Paramters
        ---------
        x : numpy.array
            The data set, a 1D numpy array, used for the construction of
            the design matrix
        """
        N = x.size
        P = self.degree
        self.matrix = np.zeros(shape=(N,P+1))
        self.matrix[:,0] = 1.0
        for j in range(1,P+1) :
            self.matrix[:,j] = x**j


    def _getMatrixFunction(self, x) :
        """Computes the design matrix for a given input function

        Computes the design matrix type function, evaluation the function
        input in the constructor at the data points given in the input. The
        first column contains unity, the subsequent columns are evaluated
        as 

            ╭     ╮      ╭   ╮
            │  X  │ =  f │ x │
            ╰   ij╯     j╰  i╯
        
        with X being the design matrix and x being the input data set. The 
        function f subscripted j is the j-th function. 
        
        Paramters
        ---------
        x : numpy.array
            The data set, a 1D numpy array, used for the construction of
            the design matrix
        """
        N = x.size
        P = self.degree
        self.matrix = np.zeros(shape=(N,P+1))
        self.matrix[:,0] = 1.0
        for i in range(N) :
            for j in range(1,P+1) :
                self.matrix[i,j] = self.model(j, x[i])











