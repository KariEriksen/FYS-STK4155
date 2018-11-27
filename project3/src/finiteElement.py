import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import fixed_quad, quad
from scipy.interpolate import lagrange

"""
class Basis :
    def __init__(self, N, dx, degree) :
        self.N       = N
        self.dx      = dx
        self.degree  = degree

        self._setupBasis()

    def _lagrangePolynomial(x, x0, j, a, b) :
        M  = self.degree
        dx = self.dx
        points = np.linspace(x0, x0+M*dx, M+1)

        xj = points[j]
        numerator = np.zeros_like(points)
        for k, xk in enumerate(points) :
            if k == j :
                numerator[k] = np.nan
            else :
                numerator[k] = xj - xk

        def polynomial(x) :
            p = 1
            for k, xk in enumerate(points) :
                if j != k :
                    p *= (x - xk) / numerator[k]
            return p 

    def _piecewise(self, x, x0, )

    def _setupBasis(self) :
        self.basisSize = self.N * self.degree
        self.basis = [None] * self.basisSize
        
        i = 0
        for j in range(self.N) :
            for d in range(self.degree) :
                self.basis[i] = lambda x, a, b: 
                i += 1



    def v(self, x, i, a, b) :
        pieces = [(x<a),
                  (x>a)*(x<b),
                  (x>b)]
        functions = [0,
                     polynomial,
                     0]
        return np.piecewise(x, pieces, functions)

N  = 5
L  = 1
x  = np.linspace(0,L,N)
dx = L/(N-1)

def PN(x, x0, dx, N, j) :
    

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
#print(A)

v = np.linalg.solve(-A,b)
#plt.plot(x,v)

u = lambda x: x*(1-x)
xx = np.linspace(0,1,100)
#plt.plot(xx, u(xx))
#plt.show()

x = np.linspace(0,10,100)
dx = 1

plt.plot(x, PN(x, 3.0, 1, 0))

plt.show()
"""



degree = 2
N = 10
dx = 1/(N-1)
points = np.linspace(0,1,N)


elements = np.zeros(shape=(N-degree,degree+1))

for i, xp in enumerate(points[:-degree]) :
    elements[i,:] = points[i:i+degree+1]


def v(x, e, j) :
    el = elements[e]
    p  = np.zeros_like(el)
    p[j] = 1
    polynomial = lagrange(el, p)
    pieces = [(x <= el[0]),
              (x >  el[0]) * (x < el[-1]),
              (x >= el[-1])]
    functions = [0,
                 polynomial,
                 0]
    return np.piecewise(x,
                        pieces,
                        functions)

def dlagrange(x, j) :
    M = len(x)

    p = np.poly1d(0.0)
    for k in range(M) :
        if k == j :
            continue

        pt = np.poly1d(1.0)
        s = x[j]-x[k]

        for i in range(M) :
            if (i == j) or (i == k) :
                continue
            denominator = x[j]-x[i]
            pt *= np.poly1d([1, -x[i]]) / denominator
        p += pt / s
    return p


def dvdx(x, e, j) :
    el = elements[e]
    polynomial = dlagrange(el, j)
    pieces = [(x <= el[0]),
              (x >  el[0]) * (x < el[-1]),
              (x >= el[-1])]
    functions = [0,
                 polynomial,
                 0]
    return np.piecewise(x,
                        pieces,
                        functions)


basisElement = []
basis  = []
for i, e in enumerate(elements) :
    for j in range(degree) :
        basis.append((i,j))
        basisElement.append(e)

basisSize = len(basis)

f = lambda x: -2

A = np.zeros(shape=(basisSize, basisSize))
b = np.zeros(shape=(basisSize))

for i in range(basisSize) :
    eli = basisElement[i]

    for j in range(basisSize) :
        if np.abs(i-j) < degree+1 :
            elj = basisElement[j]
            A[i,j] = -quad(lambda x: dvdx(x,*basis[i]) * dvdx(x,*basis[j]),
                           min(eli[0], elj[0]),
                           max(eli[-1],elj[-1]))[0]
    b[i] = quad(lambda x: f(x)*v(x,*basis[i]), 
                eli[0],
                eli[-1])[0]
b[0] = 0
b[-1] = 0
A[0,:] = 0
A[-1,:] = 0
A[0,0] = 1
A[-1,-1] = 1
c = np.linalg.solve(A,b)

N = 5000
x = np.linspace(0,1,N)
plt.plot(x,x-x**2,'k--')

u = np.zeros_like(x)
for i in range(basisSize) :
    u += c[i]*v(x, *basis[i])

print(c)
print(A)
print(b)

plt.plot(x, u, 'r-')
plt.show()











