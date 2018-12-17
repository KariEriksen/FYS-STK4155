import matplotlib.pyplot as plt
import matplotlib.transforms
import scipy.interpolate as inp
import numpy as np
import os
import sys
import scipy.linalg as scl
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
from matplotlib import cm



def FDM(Nt) :
    Nx = 11
    x_np = np.linspace(0,1,Nx)
    Nt = Nt
    t_np = np.linspace(0,0.1,Nt)
    #Finite difference schemes
    Ngrid = Nx-1 #Only inner grid points
    dx = x_np[1]-x_np[0]
    dt = (0.1-0.0) / Nt #t_np[1]-t_np[0]
    alpha = 1

    beta  = alpha**2*(dt/dx**2)
    gamma = 1+2*beta
    stable = True
    if(beta > 0.5):
        stable = False
        print("beta: %g" % beta)
        print("Explicit scheme unstable: must have beta < 0.5 for stability")

    Tfinal = 0.1
    Nsteps = int(Tfinal/dt)

    u_explicit  = np.sin(np.pi*x_np) #Initial condition
    u_implicit  = np.sin(np.pi*x_np) #Initial condition
    u_crank_nich = np.sin(np.pi*x_np) #Initial condition

    A_diag =  np.ones(Ngrid-1)*gamma
    A_off  = -beta*np.ones(Ngrid-1)
    A = sparse.diags([A_diag, A_off, A_off], offsets=[0, -1, 1]).tocsr()

    gamma1_cn = 1+beta
    gamma2_cn = 1-beta

    A_cn_diag = np.ones(Ngrid-1)*gamma1_cn
    A_cn_off = -np.ones(Ngrid-1)*0.5*beta
    B_cn_diag = np.ones(Ngrid-1)*gamma2_cn
    B_cn_off = np.ones(Ngrid-1)*0.5*beta

    A_cn = sparse.diags([A_cn_diag, A_cn_off, A_cn_off], offsets=[0, -1, 1]).tocsr()
    B_cn = sparse.diags([B_cn_diag, B_cn_off, B_cn_off], offsets=[0, -1, 1]).tocsr()

    #plt.plot(x,u_explicit,'-g',label=r'$u_e(x,0)$') #plot initial state
    tt = 0
    for t in range(1,Nt+1):
        tt += dt
        u_explicit[1:Ngrid] = u_explicit[1:Ngrid] + beta*(u_explicit[2:]-2*u_explicit[1:Ngrid]+u_explicit[0:Ngrid-1]) #Explicit Euler
        u_implicit[1:Ngrid] = sparse_linalg.spsolve(A,u_implicit[1:Ngrid]) 
        u_crank_nich[1:Ngrid] = sparse_linalg.spsolve(A_cn,B_cn.dot(u_crank_nich[1:Ngrid]))
    print(tt)
    return u_explicit, u_implicit, u_crank_nich, stable

plt.rc('text', usetex=True)
x = np.linspace(0,1,11)
ue = np.sin(np.pi*x)*np.exp(-0.1*np.pi**2)

expl, impl, crank = [],[],[]
dts = []

M = 50
first_expl = -1
for Nt in range(10,1001,M) :
    dt = 0.1/Nt
    t = 0.1
    dts.append(dt)

    expl_, impl_, crank_, stable = FDM(Nt)
    

    expl.append(np.sum(np.abs(expl_   - ue))/11)
    if not stable :
        expl[-1] = np.nan
    impl.append(np.sum(np.abs(impl_   - ue))/11)
    crank.append(np.sum(np.abs(crank_ - ue))/11)
    print(Nt, expl[-1], impl[-1], crank[-1])

skip = np.sum(~np.isfinite(expl))
print(skip)
print(expl[:skip])

expl_x = np.asarray(dts[skip:])
#plt.loglog(expl_x, expl[skip:],  'b-x', label=r'FDM-Explicit')
#plt.loglog(dts, impl,  'k-o', label=r'FDM-Implicit')
#plt.loglog(dts, crank, 'c-s', label=r'FDM-Crank')

errors = []
dts = []

for Nt in range(10,1001,5) :
    dt = 0.1/Nt
    t = 0.1
    dts.append(dt)
    ue = np.sin(np.pi*x)*np.exp(-t*np.pi**2)
    t = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 't%d.dat' %(Nt)))
    u = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'u%d.dat' %(Nt)))
    
    errors.append(np.mean(np.abs(u[-1,:]-ue)))


###
dnn_11 = np.array([  0.00000000e+0 , 1.16299335e-1 ,  2.18211440e-1,   3.01040738e-1 ,
   3.53491286e-1 ,  3.69497491e-1 ,  3.50736761e-1,   2.99996583e-1,
   2.19906798e-1 ,  1.16277424e-1 ,  1.10218212e-16])
dnn_51 = np.array(
[  0.00000000e+00 ,  1.17311885e-01 ,  2.19934974e-01,   3.03348916e-01,
   3.56145177e-01 ,  3.71937380e-01 ,  3.52557235e-01,   3.01208737e-01,
   2.20701238e-01 ,  1.16739716e-01 ,  1.10218212e-16,]
)
dnn_101 = np.array(
[  0.00000000e+00 , 1.17439073e-01 ,  2.20152434e-01,   3.03640933e-01,
   3.56485645e-01 , 3.72252585e-01 ,  3.52787570e-01,   3.01353368e-01,
   2.20789818e-01 , 1.16790173e-01 ,  1.10218212e-16]
)
dnn_x = [11, 51, 101] 
dnn   = [np.mean(np.abs(dnn_11)), np.mean(np.abs(dnn_51)), np.mean(np.abs(dnn_101))]

###

plt.loglog(0.1/np.asarray(dnn_x), dnn, 'y-^', label=r'NN')

#plt.semilogy(range(10,1001,5), errors, 'r-', label=r'FEM')
plt.loglog(dts, errors, 'r-', label=r'FEM')
plt.xlabel(r'Time step, $\Delta t$', fontsize=16)
plt.ylabel(r'Mean $|u_e(x,t=0.1) - u(x,t=0.1)|$', fontsize=16)
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'NT_error_compare.png'), transparent=True, bbox_inches='tight')
plt.show()


"""


def L(x, y) :
    return inp.lagrange(x, y)

N = 1000
E1 = np.linspace(-3,-1, N)
E2 = np.linspace(-1, 1, N)
E3 = np.linspace( 1, 3, N)

# P1 elements
L1 = L([-3,-1], [0, 1])
L2 = L([-1, 1], [1, 0])
L3 = L([-1, 1], [0, 1])
L4 = L([ 1, 3], [1, 0])


plt.rc('text', usetex=True)

fig, ax = plt.subplots()
newax = ax.twiny()

# Make some room at the bottom
fig.subplots_adjust(bottom=0.20)

# I'm guessing you want them both on the bottom...
newax.set_frame_on(True)
newax.patch.set_visible(False)
newax.xaxis.set_ticks_position('bottom')
newax.xaxis.set_label_position('bottom')
newax.spines['bottom'].set_position(('outward', 40))

#ax.set_xlabel('Red Thing')
#newax.set_xlabel('Green Thing')

ax.plot([-10, -1], [0, 0], 'b-')
ax.plot([-10, -3], [0, 0], 'r-')
ax.plot(E1, L1(E1), 'r-', label=r'$\psi_{-1}$')
ax.plot(E2, L2(E2), 'r-')
ax.plot([1, 10], [0, 0], 'r-')

ax.plot(E2, L3(E2), 'b-', label=r'$\psi_{1}$')
ax.plot(E3, L4(E3), 'b-')
ax.plot([3, 10], [0, 0], 'b-')

ax.plot([-1,-1], [-10, 10], 'k--')
ax.plot([ 1, 1], [-10, 10], 'k--')
ax.plot([-3,-3], [-10, 10], 'y--')
ax.plot([ 3, 3], [-10, 10], 'g--')
plt.rc('legend', fontsize=16)
ax.legend()

ax.set_yticks([0,1])
ax.set_xticks([-3,-1,0,1,3])
newax.set_xticks([-3,-1,1,3])
newax.set_xticklabels(['E${}_{-1}$','E${}_{0}$','E${}_{1}$'], fontsize=16)
ax.axis([-3.5, 3.5, -0.2, 1.2])
newax.axis([-3.5, 3.5, -0.2, 1.2])


dx = 1
dy = -0.1
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

# apply offset transform to all x ticklabels.
for label in newax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'P1_basis.png'), transparent=True, bbox_inches='tight')
plt.show()

"""
####################################
####################################
####################################
####################################
# P2 elements
"""
L1 = L([-3,-2,-1], [0, 0, 1])
L2 = L([-1, 0, 1], [1, 0, 0])

L5 = L([-1, 0, 1], [0, 1, 0])

L3 = L([-1, 0, 1], [0, 0, 1])
L4 = L([ 1, 2, 3], [1, 0, 0])



fig, ax = plt.subplots()
newax = ax.twiny()

# Make some room at the bottom
fig.subplots_adjust(bottom=0.20)

# I'm guessing you want them both on the bottom...
newax.set_frame_on(True)
newax.patch.set_visible(False)
newax.xaxis.set_ticks_position('bottom')
newax.xaxis.set_label_position('bottom')
newax.spines['bottom'].set_position(('outward', 40))

#ax.set_xlabel('Red Thing')
#newax.set_xlabel('Green Thing')

ax.plot([-10, -1], [0, 0], 'b-')
ax.plot([-10, -3], [0, 0], 'r-')
ax.plot(E1, L1(E1), 'r-', label=r'$\psi_{-1}$')
ax.plot(E2, L2(E2), 'r-')
ax.plot([1, 10], [0, 0], 'r-')

ax.plot(E2, L5(E2), 'k-', label=r'$\psi_{0}$')

ax.plot(E2, L3(E2), 'b-', label=r'$\psi_{1}$')
ax.plot(E3, L4(E3), 'b-')
ax.plot([3, 10], [0, 0], 'b-')


ax.plot([-1,-1], [-10, 10], 'k--')
ax.plot([ 1, 1], [-10, 10], 'k--')
ax.plot([-3,-3], [-10, 10], 'y--')
ax.plot([ 3, 3], [-10, 10], 'g--')
plt.rc('legend', fontsize=16)
ax.legend()

ax.set_yticks([0,1])
ax.set_xticks([-3,-1,0,1,3])
newax.set_xticks([-3,-1,1,3])
newax.set_xticklabels(['E${}_{-1}$','E${}_{0}$','E${}_{1}$'], fontsize=16)
ax.axis([-3.5, 3.5, -0.2, 1.2])
newax.axis([-3.5, 3.5, -0.2, 1.2])


dx = 1
dy = -0.1
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

# apply offset transform to all x ticklabels.
for label in newax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'P2_basis.png'), transparent=True, bbox_inches='tight')
plt.show()
"""

"""
####################################
####################################
####################################
####################################
E1 = np.linspace(-3, 0, N)
E2 = np.linspace( 0, 3, N)
E3 = np.linspace( 3, 6, N)


# P3 elements
L00 = L([-3, -2, -1, 0], [0, 0, 0, 1])
L01 = L([0, 1, 2, 3],    [1, 0, 0, 0])

L1 = L([0, 1, 2, 3],    [0, 1, 0, 0])
L2 = L([0, 1, 2, 3],    [0, 0, 1, 0])

L30 = L([0, 1, 2, 3], [0, 0, 0, 1])
L31 = L([3, 4, 5, 6], [1, 0, 0, 0])


fig, ax = plt.subplots()
newax = ax.twiny()

# Make some room at the bottom
fig.subplots_adjust(bottom=0.20)

# I'm guessing you want them both on the bottom...
newax.set_frame_on(True)
newax.patch.set_visible(False)
newax.xaxis.set_ticks_position('bottom')
newax.xaxis.set_label_position('bottom')
newax.spines['bottom'].set_position(('outward', 40))

ax.plot([-10, 0], [0, 0], 'b-')
ax.plot([-10, -3], [0, 0], 'r-')
ax.plot(E1, L00(E1), 'r-', label=r'$\psi_{0}$')
ax.plot(E2, L01(E2), 'r-')
ax.plot([3, 10], [0, 0], 'r-')

ax.plot(E2, L1(E2), 'k-', label=r'$\psi_1$')
ax.plot(E2, L2(E2), 'c-', label=r'$\psi_2$')

ax.plot(E2, L30(E2), 'b-', label=r'$\psi_3$')
ax.plot(E3, L31(E3), 'b-')
ax.plot([6, 10], [0, 0], 'b-')

plt.rc('legend', fontsize=16)
ax.legend()

ax.set_yticks([0,1])
ax.set_xticks([-3,0,1,2,3, 6])
newax.set_xticks([-3,0,3,6])
newax.set_xticklabels(['E${}_{-1}$','E${}_{0}$','E${}_{1}$'], fontsize=16)
ax.axis([-3.5, 6.5, -0.4, 1.2])
newax.axis([-3.5, 6.5, -0.4, 1.2])


dx = 0.8
dy = -0.1
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

# apply offset transform to all x ticklabels.
for label in newax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'P3_basis.png'), transparent=True, bbox_inches='tight')
plt.show()
"""