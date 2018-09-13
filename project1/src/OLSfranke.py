import numpy as np
from franke import franke

N = 10

x = np.random.rand(N)
y = np.random.rand(N)

X = np.c_[np.ones(N),
		  x,y,
		  x**2,x*y,y**2,
		  x**3,x**2*y,x*y**2,y**3,
		  x**4,x**3*y,x**2*y**2,x*y**3,y**4,
		  x**5,x**4*y,x**3*y**2,x**2*y**3,x*y**4,y**5]

XT_X = np.linalg.inv(np.dot(X.T,X))
XT_y = np.dot(X.T,y)
beta = np.dot(XT_X,XT_y)
print(beta)
