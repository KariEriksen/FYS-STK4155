import numpy as np
from numba import jit


@jit(nopython=True)
def computeMatrix(matrix, x, degree) :
    ind = 1
    for i in range(1,degree+1) :
        for j in range(i+1) :
            matrix[:,ind] = x[:,0]**(i-j) * x[:,1]**j
            ind += 1



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
        is not 'polynomial', or 'polynomial2D' a ValueError is raised. 
        If 'polynomial' or 'polynomial2D' is provided as the model argument 
        but the degree argument is not present, a ValueError is also raised. 
        If the degree argument is present but not an integer, a TypeError 
        is raised.

        Parameters
        ----------
        model 
            Either a callable object or a string. In the latter case, only
            'polynomial' or 'polynomial2D' is accepted.
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
        self.p2D    = {}

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
                elif model == 'polynomial2D' :
                    self.type_ = 'polynomial2D'
                    if degree != None :
                        if type(degree) == int :
                            self.degree = degree
                        else :
                            raise TypeError("The provided polynomial degree is not an integer.")
                    else :
                        raise ValueError("Using the 'polynomial2D' type DesignMatrix requires providing a polynomial degree.")
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
        specified order is setup. If the 'polynomial2D' keyword was given,
        a polynomial design matrix in 2D, including all combined 
        polynomials up to and including self.degree is setup.

        In the former case, the self._getMatrixFunction method is called,
        while the latter results in a self._getMatrixPolynomial call. If 
        the 'polynomial2D' keyword is specified, the method
        self._getMatrixPolynomial2D is called. In this case, the input
        parameter x must be a 2D numpy array.

        Paramters
        ---------
        x : numpy.array
            The data set, a 1D numpy array, used for the construction of
            the design matrix. If the 'polynomial2D' keyword is specified
            in the constructor, a 2D numpy array is expected.

        Returns
        -------
        self.matrix : numpy.array
            The design matrix, a 2D numpy array
        """
        if self.type_ == 'polynomial' :
            self._getMatrixPolynomial(x)
        elif self.type_ == 'polynomial2D' :
            self._getMatrixPolynomial2D(x)
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


    def _getMatrixPolynomial2D(self, x) :
        """Computes the design matrix for a 2D polynomial of a given combined degree

        Computes the design matrix of 2D polynomial type with degree specified 
        in the input to the constructor. The first column contains unity, 
        the subsequent columns are evaluated as 

            ╭     ╮      ╭        ╮ 
            │  X  │ =  p │ x , y  │ 
            ╰   ij╯     j╰  i   i ╯ 
        
        with X being the design matrix and (x, y) being the 2D input data set. 
        The polynomial p is the j-th polynomial in the set of linearly 
        independent combined polynomials 
                       2    2         3    3    2        2    4    4
            x ,  y ,  x ,  y , xy ,  x ,  y ,  x y ,  x y ,  x ,  y , ...

        of combined total degree self.degree. There are n(n+3)/2 such 
        different monomials/polynomials in a set of degree n.
        
        Paramters
        ---------
        x : numpy.array
            The data set, a 2D numpy array, used for the construction of
            the design matrix
        """
        """
        self.p2D = {
            # Degree 1
            0:  lambda x,y : x,
            1:  lambda x,y : y,

            # Degree 2
            2:  lambda x,y : x**2        ,
            3:  lambda x,y : x    * y    ,
            4:  lambda x,y :        y**2 ,

            # Degree 3
            5:  lambda x,y : x**3        ,
            6:  lambda x,y : x**2 * y    ,   
            7:  lambda x,y : x    * y**2 ,
            8:  lambda x,y :        y**3 ,

            # Degree 4
            9:  lambda x,y : x**4        ,
            10: lambda x,y : x**3 * y    ,
            11: lambda x,y : x**2 * y**2 ,
            12: lambda x,y : x    * y**3 ,
            13: lambda x,y :        y**4,

            # Degree 5
            14: lambda x,y : x**5        ,
            15: lambda x,y : x**4 * y    ,
            16: lambda x,y : x**3 * y**2 ,
            17: lambda x,y : x**2 * y**3 ,
            18: lambda x,y : x    * y**4 ,
            19: lambda x,y :        y**5 ,

            # Degree 6
            20: lambda x,y : x**6        ,
            21: lambda x,y : x**5 * y    ,
            22: lambda x,y : x**4 * y**2 ,
            23: lambda x,y : x**3 * y**3 ,
            24: lambda x,y : x**2 * y**4 ,
            25: lambda x,y : x   *  y**5 ,
            26: lambda x,y :        y**6 ,

            # Degree 7
            27: lambda x,y : x**7        ,
            28: lambda x,y : x**6 * y    ,
            29: lambda x,y : x**5 * y**2 ,
            30: lambda x,y : x**4 * y**3 ,
            31: lambda x,y : x**3 * y**4 ,
            32: lambda x,y : x**2 * y**5 ,
            33: lambda x,y : x    * y**6 ,
            34: lambda x,y :        y**7 ,

            # Degree 8
            35: lambda x,y : x**8        ,
            36: lambda x,y : x**7 * y    ,
            37: lambda x,y : x**6 * y**2 ,
            38: lambda x,y : x**5 * y**3 ,
            39: lambda x,y : x**4 * y**4 ,
            40: lambda x,y : x**3 * y**5 ,
            41: lambda x,y : x**2 * y**6 ,
            42: lambda x,y : x    * y**7 ,
            43: lambda x,y :        y**8 ,

            # Degree 9
            44: lambda x,y : x**9        ,
            45: lambda x,y : x**8 * y    ,
            46: lambda x,y : x**7 * y**2 ,
            47: lambda x,y : x**6 * y**3 ,
            48: lambda x,y : x**5 * y**4 ,
            49: lambda x,y : x**4 * y**5 ,
            50: lambda x,y : x**3 * y**6 ,
            51: lambda x,y : x**2 * y**7 ,
            52: lambda x,y : x    * y**8 ,
            53: lambda x,y :        y**9 ,

            # Degree 10
            54: lambda x,y : x**10       ,
            55: lambda x,y : x**9 * y    ,
            56: lambda x,y : x**8 * y**2 ,
            57: lambda x,y : x**7 * y**3 ,
            58: lambda x,y : x**6 * y**4 ,
            59: lambda x,y : x**5 * y**5 ,
            60: lambda x,y : x**4 * y**6 ,
            61: lambda x,y : x**3 * y**7 ,
            62: lambda x,y : x**2 * y**8 ,
            63: lambda x,y : x    * y**9 ,
            64: lambda x,y :        y**10
        }
        """
        N = x.shape
        N = N[0]

        # The total number of polynomials up to and including combined 
        # total degree self.degree, i.e. 2, 5, 9, 14, 20, 27... . This
        # is just the sum of the integers, 
        #
        #     n+1         n (n + 3)
        #      Σ   n   =  ─────────  =  P
        #     k=2             2          n
        #
        P = int(self.degree*(self.degree+3)/2)

        self.matrix = np.zeros(shape=(N,P+1))
        self.matrix[:,0] = 1.0
        computeMatrix(self.matrix, x, self.degree)



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
        function f subscripted j is the j-th function. The self.degree 
        value given as a parameter to the constructor dictates how many 
        different functions are used. 

        The input function must take two arguments, a number specifying 
        which function index and a numpy.array of values on which to 
        evaluate the function,

                   ╭     → ╮
            f  =  f│ j , x │
                   ╰       ╯
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
            self.matrix[:,j] = self.model(j, x)











