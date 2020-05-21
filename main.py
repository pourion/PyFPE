from solver import rbf_fp
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sympy import *

dim = 1
Ndim = 60 
box = {'xmin':-0.5, 'xmax':1.5, 'ymin':-1, 'ymax':1, 'zmin':-1, 'zmax': 1}
tf = 0.2
dt = 0.01
"""
Define PDE system here:
"""
def drift(t, x, y=None, z=None):
	if dim==1:
		return np.array([cos(2*x)])
	if dim==2:
		if y is None: x, y = x[0], x[1]
		return np.array([x, y])
	if dim==3:
		if y is None: x, y, z = x[0], x[1], x[2]
		return np.array([x, y, z])

def diffusion(t, x, y=None, z=None):
	if dim==1:
		return np.array([x*x - 0.18*x + 0.01])
	if dim==2:
		if y is None: x, y = x[0], x[1]
		return np.array([[x*x, 0.0*x], [0.0*x, y*y]])
	if dim==3:
		if y is None: x, y, z = x[0], x[1], x[2]
		return np.array([[x*x, 0.0*x, 0.0*x], [0.0*x, y*y, 0.0*x], [0.0*x, 0.0*x, z*z]])

def source(t, x, y=None, z=None):
		if dim==1:
			return cosh(x)
		if dim==2:
			if y is None: x, y = x[0], x[1]
			return 0.0
		if dim==3:
			if y is None: x, y, z = x[0], x[1], x[2]
			return 0.0

def Dirichlet_bc(t, x, y=None, z=None):
		return 1 + t

# RBF function: Multi-quadratics
def phi(x, xj, cj):
		if dim==1:
			phi = ((x - xj)**2 + cj**2)**0.5
		return phi
"""
End of PDE System
"""
self = rbf_fp(dim, box, Ndim, tf, dt, drift, diffusion, source, phi, Dirichlet_bc, True)

pdb.set_trace()