from sympy import *
init_printing(use_unicode=True)
import sympy.plotting as plt

x, xi, xj, xk = symbols('x x_i x_j x_k')

# -1 --- 0 --- 1
#  i --- j --- k
#xi = -1
#xj =  0
#xk =  1

# Langrange interpolating polynomials
Pi = (x-xj)*(x-xk) / ((xi-xj)*(xi-xk))
Pj = (x-xi)*(x-xk) / ((xj-xi)*(xj-xk))
Pk = (x-xi)*(x-xj) / ((xk-xi)*(xk-xj))

M = Matrix([[0 for i in range(3)] for j in range(3)])
K = Matrix([[0 for i in range(3)] for j in range(3)])

for i, p0 in enumerate([Pi, Pj, Pk]) :
	for j, p1 in enumerate([Pi, Pj, Pk]) :
		M[i,j] = integrate(p0*p1, (x, xi, xk))
		K[i,j] = integrate(diff(p0,x)*diff(p1,x), (x, xi, xk))

print("M: ")
dx = 0.125
pprint(M.subs({xi:0.75,xj:0.875,xk:1}))

print("K: ")
pprint(K.subs({xi:0.75,xj:0.875,xk:1}))
