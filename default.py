from solver import rbf_fp
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import scipy.integrate as integrate


class default(rbf_fp):
    def __init__(self, dim, box, Ndim, tf, dt):
        self.dim = dim
        self.box = box
        self.Ndim = Ndim
        self.tf = tf
        self.dt = dt
        self.rbf = rbf_fp(self.dim, self.box, self.Ndim, self.tf, self.dt, self.drift,
                          self.diffusion, self.source, self.phi, self.Dirichlet_bc, True)
        self.rbf.solve()

    ##############################################
    # Define PDE system here:
    ##############################################
    def drift(self, t, x, y=None, z=None):
        if dim == 1:
            return np.array([0.1 * x + 1.0])
        if dim == 2:
            if y is None:
                x, y = x[0], x[1]
            return np.array([x, y])
        if dim == 3:
            if y is None:
                x, y, z = x[0], x[1], x[2]
            return np.array([x, y, z])

    def diffusion(self, t, x, y=None, z=None):
        if dim == 1:
            return np.array([1.0 * x * x - 1.0 * x + 0.5])
        if dim == 2:
            if y is None:
                x, y = x[0], x[1]
            return np.array([[x * x, 0.0 * x], [0.0 * x, y * y]])
        if dim == 3:
            if y is None:
                x, y, z = x[0], x[1], x[2]
            return np.array([[x * x, 0.0 * x, 0.0 * x], [0.0 * x, y * y, 0.0 * x], [0.0 * x, 0.0 * x, z * z]])

    def source(self, t, x, y=None, z=None):
        if dim == 1:
            return 0 * x
        if dim == 2:
            if y is None:
                x, y = x[0], x[1]
            return 0.0
        if dim == 3:
            if y is None:
                x, y, z = x[0], x[1], x[2]
            return 0.0

    def Dirichlet_bc(self, t, x, y=None, z=None):
        return 0.0

    # RBF function: Multi-quadratics
    def phi(self, x, xj, cj):
        if dim == 1:
            phi = ((x - xj)**2 + cj**2)**0.5
        return phi


dim = 1
Ndim = 80
box = {'xmin': -1, 'xmax': 2, 'ymin': -1, 'ymax': 1, 'zmin': -1, 'zmax': 1}
tf = 1e-1
dt = 1e-3


ep = default(dim, box, Ndim, tf, dt)
pdb.set_trace()
