import os
import sys
import pytest
import numpy as np
import random
import matplotlib.pyplot as plt

# Add the src/ directory to the python path so we can import the code 
# we need to test directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from leastSquares import LeastSquares



def test_LeastSquares_fit() :
    """Tests the fit method of the Least Squares class

    The tests comprise fitting of models to known data.
    """

    # Ensure fitting polynomials of order 1 through 5 to y(x) = x results
    # in the beta corresponding to the x term equal to 1.0 and all other
    # beta values zero.
    #
    # Secondly we test on y(x)=x + 2 to also make sure the intercept is 
    # calculated correctly.
    for intercept in [0, 2] :
        for method in ['manual', 'skl'] :
            N = 10
            x = np.linspace(0,1,N)
            y = x + intercept

            for i in range(2,5+1) :
                P = i
                X = np.zeros(shape=(N,P))
                X[:,0] = 1.0

                for j in range(1,P) :
                    X[:,j] = x**j

                OLS = LeastSquares(backend=method)
                beta = OLS.fit(X,y)
                
                assert beta[0] == pytest.approx(intercept, abs=1e-10)
                assert beta[1] == pytest.approx(1.0, abs=1e-10)
                for j in range(2,P) :
                    assert beta[j] == pytest.approx(0.0, abs=1e-10)



    # Ensure the backend='manual' and the backend='skl' versions of 
    # LeastSquares.fit(X,y) give the same result.
    N = 5
    P = 5
    x = np.linspace(0,1,N)
    y = x + x**2 - (1.0 - x)**5
    
    X = np.zeros(shape=(N,P))
    X[:,0] = 1.0
    for j in range(1,P) :
        X[:,j] = x**j

    OLS = LeastSquares(backend='manual')
    beta_manual = OLS.fit(X,y)

    OLS = LeastSquares(backend='skl')
    beta_skl    = OLS.fit(X,y)

    assert beta_manual == pytest.approx(beta_skl, abs=1e-10)

def test_LeastSquares_predict() :
    """Tests the predict method of the Least Squares class

    The test is done with a known beta array, comparing results to a known
    MSE value.
    """
    N = 5
    P = 3
    x = np.linspace(0,1,N)
    random.seed(10)
    y = 3*x**2 - 9*x - 2.4*x**5 - 3.1
    X = np.zeros(shape=(N,P))
    X[:,0] = 1.0

    for j in range(1,P) :
        X[:,j] = x**j

    OLS = LeastSquares(backend='skl')
    beta_skl    = OLS.fit(X,y)
    predict_skl = OLS.predict()

    OLS = LeastSquares(backend='manual')
    beta_manual    = OLS.fit(X,y)

    # Ensure the exact same beta value are used by both backend versions.
    OLS.beta       = beta_skl
    predict_manual = OLS.predict()

    assert (predict_manual == pytest.approx(predict_skl, abs=1e-15))


def test_LeastSquares_meanSquaredError() :
    """Tests the meanSquaredError method of the Least Squares class

    The test is done with a known beta array, comparing results to a known
    MSE value.
    """
    N = 5
    P = 3
    x = np.linspace(0,1,N)
    random.seed(10)
    y = 3*x**2 - 9*x - 2.4*x**5 + 3.1
    X = np.zeros(shape=(N,P))
    X[:,0] = 1.0

    for j in range(1,P) :
        X[:,j] = x**j

    OLS = LeastSquares(backend='skl')
    beta_skl    = OLS.fit(X,y)
    MSE_skl     = OLS.meanSquaredError()

    OLS = LeastSquares(backend='manual')
    beta_manual = OLS.fit(X,y)

    # Ensure the manual and the skl fit both use the exact same beta 
    # values.
    OLS.beta    = beta_skl
    MSE_manual  = OLS.meanSquaredError()

    # beta: 
    #    2.98147321428571299151
    #   -6.48616071428570872826
    #   -1.66071428571428914012
    #
    # yHat = beta0 + beta1 x + beta2 x^2
    #    2.98147321428571299151
    #    1.25613839285714279370
    #   -0.67678571428571365765
    #   -2.81729910714285569640
    #   -5.16540178571428487686
    #
    # MSE = 1/5 * sum(yHat - y)**2
    #    0.03294015066964287725

    MSE_true = 0.03294015066964287725

    assert MSE_skl    == pytest.approx(MSE_manual, abs=1e-15)
    assert MSE_skl    == pytest.approx(MSE_true, abs=1e-15)
    assert MSE_manual == pytest.approx(MSE_true, abs=1e-15)


def test_LeastSquares_R2() :
    """Tests the R2 score method of the Least Squares class

    The test is done with a known beta array, comparing results to a known
    MSE value.
    """
    N = 5
    P = 3
    x = np.linspace(0,1,N)
    random.seed(10)
    y = 3*x**2 - 9*x - 2.4*x**5 + 3.1
    X = np.zeros(shape=(N,P))
    X[:,0] = 1.0

    for j in range(1,P) :
        X[:,j] = x**j

    OLS = LeastSquares(backend='skl')
    beta_skl   = OLS.fit(X,y)
    R2_skl     = OLS.R2()

    OLS = LeastSquares(backend='manual')
    beta_manual = OLS.fit(X,y)

    # Ensure the manual and the skl fit both use the exact same beta 
    # values.
    OLS.beta   = beta_skl
    R2_manual  = OLS.R2()

    yHat = np.dot(X, beta_skl)
    R2_true = 1.0 - np.sum((y - yHat)**2) / np.sum((y - np.mean(y))**2)
    
    # beta: 
    #    2.98147321428571299151
    #   -6.48616071428570872826
    #   -1.66071428571428914012
    #
    # yHat = beta0 + beta1 x + beta2 x^2
    #    2.98147321428571299151
    #    1.25613839285714279370
    #   -0.67678571428571365765
    #   -2.81729910714285569640
    #   -5.16540178571428487686
    #
    # y = 3x^2 - 9x -2.4x^5 + 3.1
    #    3.10000000000000008882
    #    1.03515625000000000000
    #   -0.72500000000000008882
    #   -2.53203124999999973355
    #   -5.30000000000000071054
    #
    # R2 = 1.0 - sum(yHat - y)**2 / sum(yHat - mean(y))**2
    #    0.99605957942938250227

    assert R2_skl    == pytest.approx(R2_manual, abs=1e-15)
    assert R2_skl    == pytest.approx(R2_true, abs=1e-15)
    assert R2_manual == pytest.approx(R2_true, abs=1e-15)


def test_LeastSquares_fit_ridge() :
    """Tests the fit method of the Least Squares class with method='ridge'

    The tests comprise fitting of models to known data.
    """
    N = 5
    P = 5
    x = np.linspace(0,1,N)
    y = x + x**2 - (1.0 - x)**5
    
    X = np.zeros(shape=(N,P))
    X[:,0] = 1.0
    for j in range(1,P) :
        X[:,j] = x**j

    OLS = LeastSquares(method='ols', backend='manual')
    beta_ols = OLS.fit(X,y)

    OLS = LeastSquares(method='ridge', backend='manual')
    OLS.setLambda(0.0)
    beta_lambda0 = OLS.fit(X,y)

    assert beta_lambda0 == pytest.approx(beta_ols, abs=1e-15)

    # Make sure the skl and the manual backends give the same result
    SKL = LeastSquares(method='ridge', backend='skl')
    SKL.setLambda(0.0)
    beta_skl = SKL.fit(X,y)
    
    assert beta_lambda0 == pytest.approx(beta_skl, abs=1e-10)

    SKL.setLambda = 0.5
    OLS.setLambda = 0.5
    beta_skl = SKL.fit(X,y)
    beta_lambda0 = OLS.fit(X,y)
    print(beta_lambda0)
    print(beta_skl)

    assert beta_lambda0 == pytest.approx(beta_skl, abs=1e-10)




if __name__ == '__main__':
    test_LeastSquares_R2()


