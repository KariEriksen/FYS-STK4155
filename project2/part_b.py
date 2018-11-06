import os
import sys
import numpy as np
import functools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Add the src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'project1', 'src'))

from ising          import Ising
from leastSquares   import LeastSquares


def setup(L = 40, N = 1000, train = 0.4) :
    ising   = Ising(L,N)
    ols     = LeastSquares(backend='manual', method='ols')
    ridge   = LeastSquares(backend='manual', method='ridge')
    lasso   = LeastSquares(backend='skl',    method='lasso')

    X_total, y_total = ising.generateDesignMatrix1D()

    N_train = int(train * N)
    N_test  = N - N_train

    X_train = X_total[:N_train]
    y_train = y_total[:N_train]

    X_test  = X_total[N_train:]
    y_test  = y_total[N_train:]
    
    return ising, ols, ridge, lasso, X_train, X_test, y_train, y_test, X_total, y_total   


def MSE_R2_as_function_of_lambda(L = 40, N = 1000, train=0.4, plotting=False) :
    _, ols, ridge, lasso, X_train, X_test, y_train, y_test, _, _ = setup(L,N,train)

    ols.fit(X_train, y_train)
    ols.predict(X_test)
    ols.y = y_test

    # Copy the value for convenience when plotting later.
    MSE_ols = [ols.MSE() for i in range(2)]
    R2_ols  = [ols.R2()  for i in range(2)]
    ols.y = y_train
    ols.predict(X_train)    
    R2_ols_train = [ols.R2()  for i in range(2)]

    beta_plotting = True

    M = 7
    if beta_plotting :
        cmap_args=dict(vmin=-1.0, vmax=1.0, cmap='seismic')
        fig, ax = plt.subplots(nrows=M, ncols=3)

    for j in range(M) :
        ax[j][0].imshow(ols.beta.reshape((L,L)), **cmap_args)
        ax[j][0].get_yaxis().set_ticks([])
        ax[j][0].get_xaxis().set_ticks([])
        if j==0 :
            ax[j][0].set_title('ols')

    MSE_ridge = []
    MSE_lasso = []
    R2_ridge  = []
    R2_lasso  = []
    R2_ridge_train = []
    R2_lasso_train = []

    for method in ['ridge','lasso'] :
        if method == 'ridge' :
            fitter = ridge
        elif method == 'lasso' :
            fitter = lasso

        lambdas = np.logspace(-5, 1, M)
        for j,lambda_ in enumerate(lambdas) :
            if method == 'ridge' :
                lambda_ *= 1000
            fitter.setLambda(lambda_)
            fitter.fit(X_train, y_train)
            fitter.predict(X_train)

            if method == 'ridge' :
                R2_ridge_train.append(fitter.R2())
            elif method == 'lasso' :
                R2_lasso_train.append(fitter.R2())

            fitter.predict(X_test)
            fitter.y = y_test

            if method == 'ridge' :
                MSE_ridge.append(fitter.MSE())
                R2_ridge .append(fitter.R2())
                i = 1
            elif method == 'lasso' :
                MSE_lasso.append(fitter.MSE())
                R2_lasso .append(fitter.R2())
                i = 2

            if beta_plotting :
                plt.rc('text', usetex=True)
                if method == 'lasso':
                    im = ax[j][i].imshow(fitter.beta.reshape((L,L)), **cmap_args)
                else :
                    ax[j][i].imshow(fitter.beta.reshape((L,L)), **cmap_args)
                #ax[j][i].set_title(r'$%3.3f$' %(t*N), fontsize=7)
                ax[j][i].get_yaxis().set_ticks([])
                ax[j][i].get_xaxis().set_ticks([])
                ax[j][i].set_xlabel(r'$\lambda=%f$'%lambda_)
                if j==0 :
                    ax[j][i].set_title(method)

    if beta_plotting : 
        fig = plt.gcf()
        fig.set_size_inches(7,7*np.sqrt(2), forward=True)
        """
        divider = make_axes_locatable(axarr[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=fig.colorbar(im, cax=cax)

        cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25),fontsize=14)
        cbar.set_label('$J_{i,j}$',labelpad=-40, y=1.12,fontsize=16,rotation=0)

        fig.subplots_adjust(right=2.0)
        """
        plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', '1D_ising_beta_imshow.png'), transparent=True, bbox_inches='tight')
        plt.show()

    MSE_ols   = np.array(MSE_ols)
    MSE_ridge = np.array(MSE_ridge)
    MSE_lasso = np.array(MSE_lasso)

    R2_ols   = np.array(R2_ols)
    R2_ridge = np.array(R2_ridge)
    R2_lasso = np.array(R2_lasso) 

    R2_lasso_train = np.array(R2_lasso_train)
    R2_ridge_train = np.array(R2_ridge_train)
    R2_ols_train   = np.array(R2_ols_train)   

    if plotting :
        plt.rc('text', usetex=True)
        plt.loglog([lambdas[0], lambdas[-1]], MSE_ols,   'k--',                    label=r'OLS')
        plt.loglog(lambdas,                   MSE_ridge, marker='o', markersize=2, label=r'Ridge')
        plt.loglog(lambdas,                   MSE_lasso, marker='o', markersize=2, label=r'Lasso')
        plt.legend(fontsize=10)
        plt.xlabel(r'shrinkage parameter $\lambda$', fontsize=10)
        plt.ylabel(r'MSE',                           fontsize=10)
        #plt.subplots_adjust(left=0.2,bottom=0.2)
        #plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'MSE_ising_lambda.png'), transparent=True, bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.semilogx([lambdas[0], lambdas[-1]], R2_ols,      'b--', label=r'Test  (OLS)')
        plt.semilogx([lambdas[0], lambdas[-1]], R2_ols_train,'b-',  label=r'Train (OLS)')

        plt.semilogx(lambdas, R2_ridge,       'r--', label=r'Test  (Ridge)')
        plt.semilogx(lambdas, R2_ridge_train, 'r-',  label=r'Train (Ridge)')

        plt.semilogx(lambdas, R2_lasso,       'g--', label=r'Test  (Lasso)')
        plt.semilogx(lambdas, R2_lasso_train, 'g-',  label=r'Train (Lasso)')

        plt.legend(fontsize=10)
        plt.xlabel(r'shrinkage parameter $\lambda$', fontsize=10)
        plt.ylabel(r'$R^2$ score',                   fontsize=10)
        #plt.subplots_adjust(left=0.2,bottom=0.2)
        #plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'R2_ising_lambda.png'), transparent=True, bbox_inches='tight')
        plt.show()


def MSE_R2_as_function_of_training_set_size(L=40, N=1500, M=25, plotting=False) :
    K            = 5
    train        = np.linspace(700, 900, M)
    train       /= N
    MSE          = np.zeros((M,K))
    R2           = np.zeros((M,K))
    MSE_variance = np.zeros(M)
    R2_variance  = np.zeros(M)
    
    beta_plotting = True

    if beta_plotting :
        cmap_args=dict(vmin=-1.0, vmax=1.0, cmap='seismic')
        fig, ax = plt.subplots(nrows=int(np.sqrt(M)), ncols=int(np.sqrt(M)))
        i = 0
        j = 0

    for k in range(M) :
        for it in range(K) :
            np.random.seed(1975*k + 29*it)
            #np.random.seed()
            t = train[k]
            _, ols, _, _, X_train, X_test, y_train, y_test, _, _ = setup(L, N, t)
            beta = ols.fit(X_train, y_train)
            ols.predict(X_test)
            ols.y = y_test
            MSE[k,it] = ols.MSE()
            R2 [k,it] = ols.R2()

        MSE_variance = np.var(MSE[k,:])
        R2_variance  = np.var(R2 [k,:])
        MSE[k,0] = np.mean(MSE[k,:])
        R2 [k,0] = np.mean(R2 [k,:])

        if beta_plotting :
            plt.rc('text', usetex=True)
            ax[j][i].imshow(beta.reshape((L,L)), **cmap_args)
            ax[j][i].set_title(r'$%3.3f$' %(t*N), fontsize=7)
            ax[j][i].get_yaxis().set_ticks([])
            ax[j][i].get_xaxis().set_ticks([])

            if i == int(np.sqrt(M))-1 :
                j += 1
                i  = 0
            else :
                i += 1

    if beta_plotting : 
        fig = plt.gcf()
        fig.set_size_inches(12,12)
        plt.show()

    if plotting :
        plt.rc('text', usetex=True)

        plt.errorbar(train*N, MSE[:,0], 
                        yerr        = np.sqrt(MSE_variance), 
                        fmt         = '-o',
                        markersize  = 2,
                        linewidth   = 1,
                        elinewidth  = 0.5,
                        capsize     = 2,
                        capthick    = 0.5,
                        label       = r"MSE")
        """plt.errorbar(train*N, R2[:,0], 
                        yerr        = np.sqrt(R2_variance), 
                        fmt         = '-o',
                        markersize  = 2,
                        linewidth   = 1,
                        elinewidth  = 0.5,
                        capsize     = 2,
                        capthick    = 0.5,
                        label       = r"R2")
        """
        plt.gcf().axes[0].set_yscale("log")
        plt.legend(fontsize=10)
        plt.xlabel(r'training data size',    fontsize=10)
        plt.ylabel(r'MSE',                   fontsize=10)
        plt.subplots_adjust(left=0.2,bottom=0.2)
        #plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'MSE_R2_training_size_ols.png'), transparent=True, bbox_inches='tight')
        plt.show()


def cross_validation(L = 40, N = 1000, k=10, plotting=False) :
    N = N-N%k
    N_fold = N // k
    N = N*(k+1)//k
    _, ols, ridge, lasso, _, _, _, _, X, yy = setup(L=L, N=N, train=0.5)
    _, _, _, _, _, _, _, _, X_test, yy_test = setup(L=L, N=N, train=0.5)

    ridge.setLambda(10)
    lasso.setLambda(1e-3)

    for fitter in [ols, ridge, lasso] :
        print(fitter.getMethod() + " ===========")
        
        M = 10
        noise = np.logspace(-5,1,M)
        performance = np.empty((M,3))

        for i in range(len(noise)) :
            noise_var = noise[i]
            test_noise = np.random.normal(0, np.sqrt(noise_var), size=yy_test.shape[0])
            y       = yy      + np.random.normal(0, np.sqrt(noise_var), size=yy.shape[0])
            y_test  = yy_test + test_noise#np.random.normal(0, np.sqrt(noise_var), size=yy_test.shape[0])
            
            y_predict = np.zeros((N,k))

            for fold in range(k) :
                mask = np.array([False]*N)
                mask[fold*N_fold:(fold+1)*N_fold] = True
                X_fold = X[mask,:]
                y_fold = y[mask]
                X_train = X[np.invert(mask),:]
                y_train = y[np.invert(mask)]

                beta = fitter.fit(X_train, y_train)
                y_predict[:,fold] = np.dot(X_test, beta)

            y_t = y_test
            y_test  = np.reshape(y_test,  (y_test.shape[0],1))
            yyy_test = np.reshape(yy_test, (yy_test.shape[0],1))

            MSE = np.mean( np.mean((y_test - y_predict)**2, axis=1, keepdims=True) )
            bias2 = np.mean( (yyy_test - np.mean(y_predict, axis=1, keepdims=True))**2 )
            bias2noise = np.mean( (y_test - np.mean(y_predict, axis=1, keepdims=True))**2 )
            variance = np.mean( np.var(y_predict, axis=1, keepdims=True) )

            # MSE = bias^2 + var(y_predict) + var(noise)
            print("MSE:                 ", MSE)
            print("bias^2:              ", bias2)
            print("bias^2(noise):       ", bias2noise)
            print("var(y_predict):      ", variance)
            print("var(noise):          ", noise_var)
            print("computed var(noise): ", np.var(test_noise))
            print("E(test_noise): ", np.mean(test_noise))
            print("MSE - [bias^2(noise) + var(y_predict)]: %10.15f"              %(MSE - (bias2noise+variance)))
            print("MSE - [bias^2        + var(y_predict) + var(noise)]: %10.15f" %(MSE - (bias2+variance+np.var(test_noise))))
            print(" ")
            #np.set_printoptions(precision=2, suppress=True)
            #print(np.reshape(beta,(L,L)))

            performance[i,0] = MSE
            performance[i,1] = bias2
            performance[i,2] = variance

        plt.loglog(noise, performance[:,0], label=r'MSE ' + fitter.getMethod())
        #plt.loglog(noise, performance[:,1], label='bias2')
        #plt.loglog(noise, performance[:,2], label='var')

    plt.rc('text', usetex=True)
    plt.xlabel(r'var$($noise$)$', fontsize=10)
    plt.ylabel(r'MSE', fontsize=10)
    plt.legend(fontsize=10)
    plt.subplots_adjust(left=0.2,bottom=0.2)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'MSE_as_function_of_noise.png'), transparent=True, bbox_inches='tight')
    plt.show()




        


if __name__ == '__main__':
    np.random.seed(2018)
    MSE_R2_as_function_of_lambda(L=40, N=1000, train=0.4, plotting=True)

    np.random.seed(2019)
    #MSE_R2_as_function_of_training_set_size(L=40, N=1500, M=9, plotting=True)

    np.random.seed(2020)
    #cross_validation(L=40, N=400, k=10, plotting=True)





