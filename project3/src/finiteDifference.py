import numpy as np
import matplotlib.pyplot as plt
import sys

def u_exact(x,t,alpha=1):
	return np.sin(np.pi*x)*np.exp(-np.pi**2*alpha**2*t)

Ngrid = 10

x  = np.linspace(0,1,Ngrid+1)
dx = x[1]-x[0]

u = np.sin(np.pi*x)

alpha = 1.0
dt    = 1e-4
beta  = alpha**2*(dt/dx**2)

if(beta > 0.5):
	print("Explicit scheme unstable: must have dt/dx**2 < 0.5 for stability")
	sys.exit(1)

u_new = np.zeros(u.shape[0])
Tfinal = 0.1
Nt     = int(Tfinal/dt)

plt.plot(x,u,'-g',label=r'$u_e(x,0)$') #plot initial state

for t in range(1,Nt+1):
	u[1:Ngrid] = u[1:Ngrid] + beta*(u[2:]-2*u[1:Ngrid]+u[0:Ngrid-1])

u_e = u_exact(x,Tfinal)

plt.plot(x,u_e,'-r',label=r'$u_e(x,T_f)$')
plt.plot(x,u,'-ob',label=r'$u_{euler}(x,T_f)$')
plt.legend()
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.show()
