import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import fixed_quad, quad


N  = 5
L  = 1
x  = np.linspace(0,L,N)
dx = L/(N-1)


def P1(x, x0) :
    pieces = [(x<x0-dx),
              (x<x0)*(x>x0-dx),
              (x>=x0)*(x<x0+dx),
              (x>x0+dx)]
    functions = [0,
                 lambda x: (x-(x0-dx))/dx,
                 lambda x: 1-(x-x0)/dx,
                 0]
    return np.piecewise(x,
                        pieces,
                        functions)

def dP1(x, x0) :
    pieces = [(x<x0-dx),
              (x<x0)*(x>x0-dx),
              (x>=x0)*(x<x0+dx),
              (x>x0+dx)]
    functions = [0,
                  1/dx,
                 -1/dx,
                 0]
    return np.piecewise(x,
                        pieces,
                        functions)    

f = lambda x: -2

b = np.zeros(N)
for i in range(1,N-1) :
    b[i] = quad(lambda t: f(t)*P1(t,x[i]), x[i]-dx, x[i]+dx)[0]

A = np.zeros(shape=(N,N))
#A = np.diag(np.ones(N)*dx) + np.diag(np.ones(N-1)*(-dx), k=1)  + np.diag(np.ones(N-1)*(-dx), k=-1)
for i in range(N) :
    for j in range(N) :
        if i==j :
            A[i,j] = quad(lambda t: dP1(t,x[i])*dP1(t,x[j]), x[i]-dx, x[i]+dx)[0]
        if (np.abs(i-j) == 1) :
            A[i,j] = quad(lambda t: dP1(t,x[i])*dP1(t,x[j]), min(x[i],x[j])-dx, max(x[i],x[j])+dx)[0]
#A[:,0]     = 0
A[0,:]     = 0
A[N-1,:]   = 0
#A[:,N-1]   = 0
A[0,0]     = 1
A[N-1,N-1] = 1
print(A)

v = np.linalg.solve(-A,b)
plt.plot(x,v)

u = lambda x: x*(1-x)
xx = np.linspace(0,1,100)
plt.plot(xx, u(xx))
plt.show()