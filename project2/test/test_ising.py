import numpy as np
import scipy 
import warnings
import os 
import sys 
import pytest

# Add the project1/src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ising import Ising


def test_ising() :
	ising = Ising()

	err = None
	try:
		s, E = ising.generateStates1D()
	except ValueError as e:
		# This should result in a value error, we catch it and ignore.
		err = e
	assert str(err) == "System size and/or number of states not specified."
	
	L = 5
	N = 10
	s, E = ising.generateStates1D(L,N)

	# Ensure the correct number of states are generated, and that they all
	# contain the correct number of spins.
	assert s.shape[0] == N
	assert s.shape[1] == L
	assert E.shape[0] == N

	# The energy can never be more than L or lower than -L.
	assert np.max(E) <= L
	assert np.min(E) >= -L

	N = 10
	L = 15
	ising.generateStates1D(L,N)
	states = np.ones(shape=(N,L))
	ising.states = states
	E = ising.computeEnergy1D()

	# All spins pointing in the same direction should yield E = -L
	assert np.all(E == np.ones(N) * (-L))

	# Flipping a single spin should give energy E = -L + 4, since 
	# two interactions which previously gave -1 energy contributions
	# now give +1.
	np.fill_diagonal(ising.states, -1)
	E = ising.computeEnergy1D()
	assert np.all(E == np.ones(N) * (-L+4))

	# Flipping another one *not* adjacent to the first flipped gives 
	# another +4 to the energy.
	np.fill_diagonal(ising.states[:,2:], -1)
	E = ising.computeEnergy1D()
	assert np.all(E == np.ones(N) * (-L+2*4))

	# Flipping one spin in between two flipped ones, i.e. 
	#
	#    -1 1 -1    -->    -1 -1 -1
	# 
	# gives -2 change in the energy.
	np.fill_diagonal(ising.states[:,1:], -1)
	E = ising.computeEnergy1D()
	assert np.all(E == np.ones(N) * (-L+2*4-4))

	L = 3
	N = 3
	ising = Ising(L,N)
	states = np.array([	[10, 200, 3000],
						[40, 500, 6000],
						[70, 800, 9000] ])
	ising.generateStates1D()
	ising.states = states 
	ising.computeEnergy1D()
	X, y = ising.generateDesignMatrix1D(L, N)
	
	# Make sure every combination of 10, 200, 3000 multiplied together
	# exists in the first row of X, every combination of 40, 500, 6000
	# is contained in the second row, etc.
	for row in range(3) :
		for i in states[row,:] :
			for j in states[row,:] :
				assert np.any(X[row,:] == i*j)
	

