import matplotlib.pyplot as plt
import matplotlib.transforms
import scipy.interpolate as inp
import numpy as np
import os
import sys



plt.rc('text', usetex=True)


x = np.linspace(0,1,11)
ue = np.sin(np.pi*x)*np.exp(-0.1*np.pi**2)

errors = []

for i in range(10,1001,5) :
    t = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 't%d.dat' %(i)))
    u = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'u%d.dat' %(i)))
    
    errors.append(np.mean(np.sum(np.abs(u[-1,:]-ue))))


plt.semilogy(range(10,1001,5), errors, 'r.', markersize=0.5)
plt.xlabel(r'Time steps, $N_t$', fontsize=16)
plt.ylabel(r'Mean $|u_e-u|$', fontsize=16)
plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'FEM_Nt.png'), transparent=True, bbox_inches='tight')
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
L1 = L([-3,-2,-1], [0, 0, 1])
L2 = L([-1, 0, 1], [1, 0, 0])

L5 = L([-1, 0, 1], [0, 1, 0])

L3 = L([-1, 0, 1], [0, 0, 1])
L4 = L([ 1, 2, 3], [1, 0, 0])


"""
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