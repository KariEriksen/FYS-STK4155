import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.linalg as scl
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

def exact_solution(x,t,alpha=1):
	return np.sin(np.pi*x)*np.exp(-np.pi**2*alpha**2*t)

Ngrid = 10
x  = np.linspace(0,1,Ngrid+1)
dx = x[1]-x[0]

alpha = 1.0
dt    = 1e-4
beta  = alpha**2*(dt/dx**2)
gamma = 1+2*beta

if(beta > 0.5):
	print("Explicit scheme unstable: must have dt/dx**2 < 0.5 for stability")
	sys.exit(1)

Tfinal = 0.1
Nt     = int(Tfinal/dt)

u_explicit  = np.sin(np.pi*x) #Initial condition
u_implicit  = np.sin(np.pi*x) #Initial condition

A_diag =  np.ones(Ngrid-1)*gamma
A_off  = -beta*np.ones(Ngrid-1)
A = sparse.diags([A_diag, A_off, A_off], offsets=[0, -1, 1]).tocsr()

#plt.plot(x,u_explicit,'-g',label=r'$u_e(x,0)$') #plot initial state

for t in range(1,Nt+1):
    u_explicit[1:Ngrid] = u_explicit[1:Ngrid] + beta*(u_explicit[2:]-2*u_explicit[1:Ngrid]+u_explicit[0:Ngrid-1]) #Explicit Euler
    u_implicit[1:Ngrid] = sparse_linalg.spsolve(A,u_implicit[1:Ngrid]) 

print(u_implicit)
u_exact = exact_solution(x,Tfinal)

plt.plot(x,u_exact,'-r',label=r'$u_{exact}(x,T_f)$')
plt.plot(x,u_explicit,'-^b',alpha=0.3,label=r'$u_{explicit}(x,T_f)$')
plt.plot(x,u_implicit,'-og',alpha=0.5,label=r'$u_{implicit}(x,T_f)$')
plt.legend()
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.show()
