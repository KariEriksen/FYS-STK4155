import numpy as np
from franke import franke

N = 100

x = np.random.rand(N)
y = np.random.rand(N)

X = np.c_[np.ones(N),
		  x,y,
		  x**2,x*y,y**2,
		  x**3,x**2*y,x*y**2,y**3,
		  x**4,x**3*y,x**2*y**2,x*y**3,y**4,
		  x**5,x**4*y,x**3*y**2,x**2*y**3,x*y**4,y**5]


z = franke(x,y)

XT_X = np.linalg.inv(np.dot(X.T,X))
XT_z = np.dot(X.T,z)
beta = np.dot(XT_X,XT_z)
print(beta)
