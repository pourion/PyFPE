from solver import rbf_fp
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sympy import *

class EP(rbf_fp):
	def __init__(self, dim, box, Ndim, tf, dt, epsilon):
		self.dim     = dim
		self.box     = box
		self.Ndim    = Ndim
		self.tf      = tf
		self.dt      = dt
		self.epsilon = epsilon    # correlation between alpha and gamma \in [-1, 1]
		self.set_params()
		# pdb.set_trace()
		rbf = rbf_fp(self.dim, self.box, self.Ndim, self.tf, self.dt, self.get_ep_drift, self.get_ep_diffusion, self.source, self.phi, self.Dirichlet_bc, True)
		rbf.solve()

	def set_params(self):
		self.scaling = 1e-3            
		self.E0      = 40000*self.scaling                        # V/mm
		self.sigma_e = 15*self.scaling                           # S/mm
		self.sigma_c = 1*self.scaling                            # S/mm
		self.Cm      = 0.01*self.scaling**2                      # F/mm^2
		self.SL      = 1.9*self.scaling**2                       # S/mm^2
		self.R       = 7.0e-6/self.scaling                       # mm
		self.phhi     = 0.13

		self.sigma_t = 2*self.sigma_e + self.sigma_c + self.phhi*(self.sigma_e - self.sigma_c)
		self.alpha_b = 3*self.sigma_e*self.sigma_c/(self.Cm*self.R*self.sigma_t)
		self.gamma_b = self.SL/self.Cm + (self.sigma_e * self.sigma_c * (2 + self.phhi))/(self.R*self.Cm*self.sigma_t)
		# stochastic parameters
		self.gamma_p = np.sqrt(self.gamma_b/7.0)        # noise in gamma
		self.alpha_p = np.sqrt(self.alpha_b/3.0)        # noise in alpha                          

	def get_pulse(self, t):
		return self.E0*0.01

	def get_ep_drift(self, t, x, y=None, z=None):
		return np.array([(self.gamma_b - 0.5*self.gamma_p**2)*x + (self.alpha_b - 0.5*self.epsilon*self.gamma_p*self.alpha_p)*self.get_pulse(t)])

	def get_ep_diffusion(self, t, x, y=None, z=None):
		return np.array([0.5*self.gamma_p**2*x*x + self.epsilon*self.alpha_p*self.gamma_p*self.get_pulse(t)*x + 0.5*self.alpha_p**2*self.get_pulse(t)**2])
	##############################################
	# Define PDE system here:
	##############################################
	def drift(self, t, x, y=None, z=None):
		if dim==1:
			return np.array([0*x])
		if dim==2:
			if y is None: x, y = x[0], x[1]
			return np.array([x, y])
		if dim==3:
			if y is None: x, y, z = x[0], x[1], x[2]
			return np.array([x, y, z])

	def diffusion(self, t, x, y=None, z=None):
		if dim==1:
			return np.array([0*x*x - 0.0*x + 1])
		if dim==2:
			if y is None: x, y = x[0], x[1]
			return np.array([[x*x, 0.0*x], [0.0*x, y*y]])
		if dim==3:
			if y is None: x, y, z = x[0], x[1], x[2]
			return np.array([[x*x, 0.0*x, 0.0*x], [0.0*x, y*y, 0.0*x], [0.0*x, 0.0*x, z*z]])

	def source(self, t, x, y=None, z=None):
			if dim==1:
				return 0*x
			if dim==2:
				if y is None: x, y = x[0], x[1]
				return 0.0
			if dim==3:
				if y is None: x, y, z = x[0], x[1], x[2]
				return 0.0

	def Dirichlet_bc(self, t, x, y=None, z=None):
			return 0.0

	# RBF function: Multi-quadratics
	def phi(self, x, xj, cj):
			if dim==1:
				phi = ((x - xj)**2 + cj**2)**0.5
			return phi
	

dim = 1
Ndim = 100 
box = {'xmin':-0.5, 'xmax':1.5, 'ymin':-1, 'ymax':1, 'zmin':-1, 'zmax': 1}
tf = 1e-6
dt = 1e-9
epsilon = 0.9
ep = EP(dim, box, Ndim, tf, dt, epsilon)
pdb.set_trace()

