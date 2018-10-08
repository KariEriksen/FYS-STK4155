import os
import sys
import pytest
import numba
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import functools
import time
from numba import jit
from PIL import Image

# Add the src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from franke         import franke
from designMatrix   import DesignMatrix
from leastSquares   import LeastSquares
from bootstrap      import Bootstrap


def real_data(file_number=2) :
    fileName = os.path.join(os.path.dirname(__file__),  
                            'data', 
                            'SRTM_data_Norway_' + str(file_number) + '.tif')
    image = Image.open(fileName, mode='r')
    image.mode = 'I'
    x = np.linspace(0, 1, image.size[0])
    y = np.linspace(0, 1, image.size[1])
    X,Y = np.meshgrid(x,y)
    Z = np.array(image)
    Z = Z - np.min(Z)
    Z = Z / np.max(Z)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    print(Z.shape)
    x_data = np.vstack([X.ravel(), Y.ravel()]).T
            
    # When plotting the Franke function itself, we use these lines.
    y_data = np.reshape(Z, x_data.data.shape[0])

    print(x_data.shape)

    ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(30, 45+90)
    #plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'terrain'+str(file_number)+'.png'), transparent=True, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    real_data()