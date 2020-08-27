#######################################################
## AUTHOR: POURIA A. MISTANI                         ##
## EMAIL: p.a.mistani@gmail.com                      ##
## "General Solver for Fokker-Planck Equations"      ##
#######################################################
import numpy as np
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sym
import scipy.integrate as integrate

class rbf_fp:
	"""
	params:
	@drift = function that admits coordinates, and returns drift coefficient there
	@diffusion = function that admits coordinates, and returns diffusion coefficient there
	@Ndim = number of nodes per dimensions
	@dim = dimension of the problem, dim=1,2,3
	@box = specificities of the computational box 
	"""
	def __init__(self, dim, box, Ndim, tf=1.0, dt=0.01, drift=None, diffusion=None, source=None, phi=None, Dirichlet_bc=None, time_dependent=True):
		self.PLOTMESH = False
		self.rbf = "MQ"
		self.QR  = True
		self.dt  = dt
		self.tf  = tf
		self.tn  = 0.0
		self.time_dependent = time_dependent
		self.x_mean = []
		self.x_var  = []
		self.mass_store = []
		self.dim  = dim
		self.Ndim = Ndim
		self.xmin = box['xmin']
		self.xmax = box['xmax']
		if self.dim>=2:
			self.ymin = box['ymin']
			self.ymax = box['ymax']
		if self.dim==3:
			self.zmin = box['zmin']
			self.zmax = box['zmax']
		self.mesh('uniform')
		if drift is not None:
			self.drift = drift
		else:
			self.drift = self.drift_default
		if diffusion is not None:
			self.diffusion = diffusion
		else:
			self.diffusion = self.diffusion_default
		if source is not None:
			self.source = source
		else:
			self.source = self.source_default
		if phi is not None:
			self.phix = phi
		else:
			self.phix = self.phi_default
		if Dirichlet_bc is not None:
			self.Dirichlet_bc = Dirichlet_bc
		else:
			self.Dirichlet_bc = self.Dirichlet_bc_default
		self.setup_coefficients()
		self.report_PDE()
		self.initial_condition()
		self.M_built = False
		self.build_linear_system()
		# self.solve()


	def mesh(self, type='uniform'):
		if type=='uniform':
			if self.dim==1:
				self.X = np.linspace(self.xmin, self.xmax, self.Ndim)
				self.Xm_wall = np.array([ 0])
				self.Xp_wall = np.array([self.Ndim-1])
				self.walls_idx = np.concatenate((self.Xm_wall, self.Xp_wall))
				self.interior_idx = np.arange(1, self.Ndim-1)
				self.NB = len(self.walls_idx)
				self.NI = len(self.interior_idx)
			elif self.dim==2:
				xx = np.linspace(self.xmin, self.xmax, self.Ndim)
				yy = np.linspace(self.ymin, self.ymax, self.Ndim)
				self.X, self.Y = np.meshgrid(xx, yy)
				self.Xp_wall = []
				self.Xm_wall = []
				self.Yp_wall = []
				self.Ym_wall = []
				for i in range(self.Ndim):	self.Xp_wall.append(( i, self.Ndim-1))
				for i in range(self.Ndim):	self.Xm_wall.append(( i, 0))
				for i in range(1, self.Ndim-1):	self.Yp_wall.append((self.Ndim-1, i))     # the corners are excluded 
				for i in range(1, self.Ndim-1):	self.Ym_wall.append(( 0, i))	 # to prevent double counting
				self.Xp_wall = np.array(self.Xp_wall)
				self.Xm_wall = np.array(self.Xm_wall)
				self.Yp_wall = np.array(self.Yp_wall)
				self.Ym_wall = np.array(self.Ym_wall)
				self.walls_idx = np.concatenate((self.Xm_wall, self.Xp_wall, self.Ym_wall, self.Yp_wall))
				self.interior_idx = []
				for i in range(self.Ndim):
					for j in range(self.Ndim):
						if i not in [0, self.Ndim-1] and j not in [0, self.Ndim-1]:
							self.interior_idx.append((i, j))
				self.interior_idx = np.array(self.interior_idx)
				self.NB = len(self.walls_idx)
				self.NI = len(self.interior_idx)
			elif self.dim==3:
				xx = np.linspace(self.xmin, self.xmax, self.Ndim)
				yy = np.linspace(self.ymin, self.ymax, self.Ndim)
				zz = np.linspace(self.zmin, self.zmax, self.Ndim)
				self.X, self.Y, self.Z = np.meshgrid(xx, yy, zz)
				# order: point_index = (x_index, y_index)
				self.Xp_wall = []
				self.Xm_wall = []
				self.Yp_wall = []
				self.Ym_wall = []
				self.Zp_wall = []
				self.Zm_wall = []
				for i in range(self.Ndim):	
					for j in range(self.Ndim):
						self.Xp_wall.append(( i,self.Ndim-1, j ))
				for i in range(self.Ndim):	
					for j in range(self.Ndim):
						self.Xm_wall.append(( i, 0, j ))

				for i in range(1, self.Ndim-1):	
					for j in range(self.Ndim):
						self.Yp_wall.append((self.Ndim-1, i, j))   # the corners are excluded 
				for i in range(1, self.Ndim-1):	
					for j in range(self.Ndim):
						self.Ym_wall.append(( 0, i, j))	 # to prevent double counting

				for i in range(1, self.Ndim-1):	
					for j in range(1, self.Ndim-1):
						self.Zp_wall.append(( i, j,self.Ndim-1))	 # to prevent double counting
				for i in range(1, self.Ndim-1):	
					for j in range(1, self.Ndim-1):
						self.Zm_wall.append(( i, j, 0))	 # to prevent double counting
				self.Xp_wall = np.array(self.Xp_wall)
				self.Xm_wall = np.array(self.Xm_wall)
				self.Yp_wall = np.array(self.Yp_wall)
				self.Ym_wall = np.array(self.Ym_wall)
				self.Zp_wall = np.array(self.Zp_wall)
				self.Zm_wall = np.array(self.Zm_wall)
				self.walls_idx = np.concatenate((self.Xm_wall, self.Xp_wall, self.Ym_wall, self.Yp_wall, self.Zm_wall, self.Zp_wall))
				self.interior_idx = []
				for i in range(1, self.Ndim-1):
					for j in range(1, self.Ndim-1):
						for k in range(1, self.Ndim-1):
								self.interior_idx.append((i, j, k))
				self.interior_idx = np.array(self.interior_idx)
				self.NB = len(self.walls_idx)
				self.NI = len(self.interior_idx)
			self.get_walls_internals_coords()
			if self.PLOTMESH: self.plot_mesh()
			self.N = self.NI + self.NB
			self.Delta = 0.002 #5e-2*(self.xmax - self.xmin)/self.Ndim			
		
		elif type=='halton':
			pass

	def unwrap(self, pnt):
		if self.dim==1:
			return pnt
		elif self.dim==2:
			return pnt[0], pnt[1]
		elif self.dim==3:
			return pnt[0], pnt[1], pnt[2]

	def get_point(self, node):
		if self.dim==1:
			return self.X[node]
		if self.dim==2:
			point = (node[0], node[1])
			return [self.X[point], self.Y[point]]
		if self.dim==3:
			point = (node[0], node[1], node[2])
			return [self.X[point], self.Y[point], self.Z[point]]

	def get_walls_internals_coords(self):
		self.Walls = []
		for node in self.walls_idx:
			crd = self.get_point(node)
			self.Walls.append(crd)
		self.Walls = np.array(self.Walls)
		self.Internals = []
		for node in self.interior_idx:
			crd = self.get_point(node)
			self.Internals.append(crd)
		self.Internals = np.array(self.Internals)

	def plot_mesh(self):
		if self.dim==1:
			fig = plt.figure(figsize=(7,7))
			ax = fig.add_subplot(111)
			ax.scatter(self.Walls, np.ones_like(self.Walls), color='b', s=4)
			ax.scatter(self.Internals, np.ones_like(self.Internals), color='r', s=4)
			ax.set_xlabel('X', fontsize=25)
			plt.tight_layout()
			plt.show()
		if self.dim==2:
			fig = plt.figure(figsize=(7,7))
			ax = fig.add_subplot(111)
			ax.scatter(self.Walls[:,0], self.Walls[:,1], color='b', s=4)
			ax.scatter(self.Internals[:,0], self.Internals[:,1], color='r', s=4)
			ax.set_xlabel('X', fontsize=25)
			ax.set_ylabel('Y', fontsize=25)
			plt.tight_layout()
			plt.show()	
		if self.dim==3:
			fig = plt.figure(figsize=(7,7))
			ax = fig.add_subplot(111, projection='3d')
			ax.scatter(self.Walls[:,0], self.Walls[:,1], self.Walls[:,2], color='b', s=4)
			ax.scatter(self.Internals[:,0], self.Internals[:,1], self.Internals[:,2], color='r', s=4)
			ax.set_xlabel('X', fontsize=25)
			ax.set_ylabel('Y', fontsize=25)
			ax.set_zlabel('Z', fontsize=25)
			plt.tight_layout()
			plt.show()	

#######################################################
##                Define PDE system                  ##
#######################################################
	"""
	drift coefficient 
	@pnt: [x, y, z]
	"""
	def drift_default(self, t, x, y=None, z=None):
		if self.dim==1:
			return np.array([0.85 - x])
		if self.dim==2:
			if y is None: x, y = x[0], x[1]
			return np.array([x, y])
		if self.dim==3:
			if y is None: x, y, z = x[0], x[1], x[2]
			return np.array([x, y, z])
	"""
	diffusion coefficient 
	@pnt: [x, y, z]
	"""
	def diffusion_default(self, t, x, y=None, z=None):
		if self.dim==1:
			return np.array([x*x - 0.18*x + 0.01])
		if self.dim==2:
			if y is None: x, y = x[0], x[1]
			return np.array([[x*x, 0.0*x], [0.0*x, y*y]])
		if self.dim==3:
			if y is None: x, y, z = x[0], x[1], x[2]
			return np.array([[x*x, 0.0*x, 0.0*x], [0.0*x, y*y, 0.0*x], [0.0*x, 0.0*x, z*z]])
	"""
	source term 
	@pnt: [x, y, z]
	"""
	def source_default(self, t, x, y=None, z=None):
		if self.dim==1:
			return 0.0
		if self.dim==2:
			if y is None: x, y = x[0], x[1]
			return 0.0
		if self.dim==3:
			if y is None: x, y, z = x[0], x[1], x[2]
			return 0.0

	def phi_utility(self, j):
		cmin = -0.05
		cmax = 0.05
		xj   = self.X[j]
		cj   = cmin + (cmax - cmin)*j/(self.N - 1)
		return cj, xj

	def phi_default(self, x, xj, cj):
		if self.dim==1:
			phi = ((x - xj)**2 + cj**2)**0.5
		return phi


	def report_PDE(self):
		print('PHI          = ', self.phix(self.xs, self.xjs, self.cjs))
		if self.dim==1:
			print('Drift        = ', self.drift(self.ts, self.xs))
			print('Diffusion    = ', self.diffusion(self.ts, self.xs))
			print('Source       = ', self.source(self.ts, self.xs))
			print('Dirichlet BC = ', self.Dirichlet_bc(self.ts, self.xs))
		elif self.dim==2:
			print('Drift        = ', self.drift(self.ts, self.xs, self.ys))
			print('Diffusion    = ', self.diffusion(self.ts, self.xs, self.ys))
			print('Source       = ', self.source(self.ts, self.xs, self.ys))
			print('Dirichlet BC = ', self.Dirichlet_bc(self.ts, self.xs, self.ys))
		elif self.dim==3:
			print('Drift        = ', self.drift(self.ts, self.xs, self.ys, self.zs))
			print('Diffusion    = ', self.diffusion(self.ts, self.xs, self.ys, self.zs))
			print('Source       = ', self.source(self.ts, self.xs, self.ys, self.zs))
			print('Dirichlet BC = ', self.Dirichlet_bc(self.ts, self.xs, self.ys, self.zs))
#######################################################
#       EVALUATE DERIVATIVES SYMBOLICALLY
#######################################################
	def setup_coefficients(self):
		self.ts = sym.symbols('t')
		# define symbols
		if self.dim==1:
			self.xs            = sym.symbols('x')
			self.xjs           = sym.symbols('xj')
			self.cjs           = sym.symbols('cj')
			self.sym_drift     = self.drift(self.ts, self.xs)
			self.sym_diffusion = self.diffusion(self.ts, self.xs)
			self.sym_phi       = self.phix(self.xs, self.xjs, self.cjs)

			self.dxDrift       = self.sym_drift[0].diff('x')

			self.dxDiff_xx     = self.sym_diffusion[0].diff('x')
			self.dxdxDiff_xx   = self.dxDiff_xx.diff('x')

			self.dxPhi         = self.sym_phi.diff('x')
			self.dxdxPhi       = self.dxPhi.diff('x')			
		elif self.dim==2:
			self.xs, self.ys   = sym.symbols('x'), sym.symbols('y')
			self.sym_drift     = self.drift(self.ts, self.xs, self.ys)
			self.sym_diffusion = self.diffusion(self.ts, self.xs, self.ys)
			self.dxDrift       = self.sym_drift[0].diff('x')
			self.dyDrift       = self.sym_drift[1].diff('y')
			
			self.dxDiff_xx     = self.sym_diffusion[0][0].diff('x')
			self.dxDiff_xy     = self.sym_diffusion[0][1].diff('x')
			self.dyDiff_xy     = self.sym_diffusion[0][1].diff('y')
			self.dxDiff_yx     = self.sym_diffusion[1][0].diff('x')
			self.dyDiff_yx     = self.sym_diffusion[1][0].diff('y')
			self.dyDiff_yy     = self.sym_diffusion[1][1].diff('y')

			self.dxdxDiff_xx   = self.dxDiff_xx.diff('x')
			self.dxdyDiff_xy   = self.dyDiff_xy.diff('x')
			self.dydxDiff_yx   = self.dxDiff_yx.diff('y')
			self.dydyDiff_yy   = self.dyDiff_yy.diff('y')
		elif self.dim==3:
			self.xs, self.ys   = sym.symbols('x'), sym.symbols('y'), 
			self.zs            = sym.symbols('z')
			self.sym_drift     = self.drift(self.ts, self.xs, self.ys, self.zs)
			self.sym_diffusion = self.diffusion(self.ts, self.xs, self.ys, self.zs)
			self.dxDrift       = self.sym_drift[0].diff('x')
			self.dyDrift       = self.sym_drift[1].diff('y')
			self.dzDrift       = self.sym_drift[2].diff('z')
			
			self.dxDiff_xx     = self.sym_diffusion[0][0].diff('x')
			self.dxDiff_xy     = self.sym_diffusion[0][1].diff('x')
			self.dyDiff_xy     = self.sym_diffusion[0][1].diff('y')
			self.dxDiff_xz     = self.sym_diffusion[0][2].diff('x')
			self.dzDiff_xz     = self.sym_diffusion[0][2].diff('z')
			self.dxDiff_yx     = self.sym_diffusion[1][0].diff('x')
			self.dyDiff_yx     = self.sym_diffusion[1][0].diff('y')
			self.dyDiff_yy     = self.sym_diffusion[1][1].diff('y')
			self.dyDiff_yz     = self.sym_diffusion[1][2].diff('y')
			self.dzDiff_yz     = self.sym_diffusion[1][2].diff('z')
			self.dzDiff_zx     = self.sym_diffusion[2][0].diff('z')
			self.dxDiff_zx     = self.sym_diffusion[2][0].diff('x')
			self.dzDiff_zy     = self.sym_diffusion[2][1].diff('z')
			self.dyDiff_zy     = self.sym_diffusion[2][1].diff('y')
			self.dzDiff_zz     = self.sym_diffusion[2][2].diff('z')

			self.dxdxDiff_xx   = self.dxDiff_xx.diff('x')
			self.dxdyDiff_xy   = self.dyDiff_xy.diff('x')
			self.dxdzDiff_xz   = self.dzDiff_xz.diff('x')
			self.dydxDiff_yx   = self.dxDiff_yx.diff('y')
			self.dydyDiff_yy   = self.dyDiff_yy.diff('y')
			self.dydzDiff_yz   = self.dzDiff_yz.diff('y')
			self.dzdxDiff_zx   = self.dxDiff_zx.diff('z')
			self.dzdyDiff_zy   = self.dyDiff_zy.diff('z')
			self.dzdzDiff_zz   = self.dzDiff_zz.diff('z')

	def D_drift(self, t, x, y=None, z=None):
		if self.dim==1:
			return np.array([self.dxDrift.subs({self.xs:x})], dtype='float32')
		if self.dim==2:
			if y is None: x, y = x[0], x[1]
			return np.array([self.dxDrift.subs({self.xs:x, self.ys:y}),\
							 self.dyDrift.subs({self.xs:x, self.ys:y})], dtype='float32')
		if self.dim==3:
			if y is None: x, y, z = x[0], x[1], x[2]
			return np.array([self.dxDrift.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dyDrift.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dzDrift.subs({self.xs:x, self.ys:y, self.zs:z})], dtype='float32')

	# order follows a convention that in 2D the first 3 entries have odd number of "x" indices
	# and then odd number of "y" indices. This simplifies building M matrix and G matrix.
	def D_diffusion(self, t, x, y=None, z=None):
		if self.dim==1:
			return np.array([self.dxDiff_xx.subs({self.xs:x})], dtype='float32')
		elif self.dim==2:
			if y is None: x, y = x[0], x[1]
			return np.array([self.dxDiff_xx.subs({self.xs:x, self.ys:y}), \
							 self.dyDiff_xy.subs({self.xs:x, self.ys:y}), \
							 self.dyDiff_yx.subs({self.xs:x, self.ys:y}), \
							 self.dxDiff_xy.subs({self.xs:x, self.ys:y}), \
							 self.dxDiff_yx.subs({self.xs:x, self.ys:y}), \
							 self.dyDiff_yy.subs({self.xs:x, self.ys:y}) ], dtype='float32')
		elif self.dim==3:
			if y is None: x, y, z = x[0], x[1], x[2]
			return np.array([self.dxDiff_xx.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dyDiff_xy.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dzDiff_xz.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dyDiff_yx.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dzDiff_zx.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dxDiff_xy.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dxDiff_yx.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dyDiff_yy.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dzDiff_yz.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dzDiff_zy.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dxDiff_xz.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dyDiff_yz.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dxDiff_zx.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dyDiff_zy.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dzDiff_zz.subs({self.xs:x, self.ys:y, self.zs:z}) ], dtype='float32')

	def DD_diffusion(self, t, x, y=None, z=None):
		if self.dim==1:
			return np.array([self.dxdxDiff_xx.subs({self.xs:x})], dtype='float32')
		elif self.dim==2:
			if y is None: x, y = x[0], x[1]
			return np.array([self.dxdxDiff_xx.subs({self.xs:x, self.ys:y}), \
							 self.dxdyDiff_xy.subs({self.xs:x, self.ys:y}), \
							 self.dydxDiff_yx.subs({self.xs:x, self.ys:y}), \
							 self.dydyDiff_yy.subs({self.xs:x, self.ys:y})], dtype='float32')
		elif self.dim==3:
			if y is None: x, y, z = x[0], x[1], x[2]
			return np.array([self.dxdxDiff_xx.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dxdyDiff_xy.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dxdzDiff_xz.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dydxDiff_yx.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dydyDiff_yy.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dydzDiff_yz.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dzdxDiff_zx.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dzdyDiff_zy.subs({self.xs:x, self.ys:y, self.zs:z}), \
							 self.dzdzDiff_zz .subs({self.xs:x, self.ys:y, self.zs:z})], dtype='float32')

	def phi_s(self, xi, j):
		if self.dim==1:
			cj, xj = self.phi_utility(xi, j)
			ph     = self.phix(xi, xj, cj)
		return ph

	def D_phi_s(self, xi, j):
		if self.dim==1:
			cj, xj = self.phi_utility(xi, j)
			dph    = self.dxPhi.subs({'x':1, 'xj': 0, 'cj':1})
		return dph

	def DD_phi_s(self, xi, j):
		if self.dim==1:
			cj, xj = self.phi_utility(xi, j)
			ddph   = self.dxdxPhi.subs({'x':1, 'xj': 0, 'cj':1})
		return ddph


#######################################
##	Initial and Boundary conditions
#######################################
	"""
	initialize by delta distribution around origin
	see Dehghan & Mohammadi 2014
	"""
	def initial_condition(self):
		self.Pn = (0.5/np.sqrt(np.pi*self.Delta))*np.exp(-self.X**2/(4.0*self.Delta))
		if self.dim > 1:
			self.Pn *= (0.5/np.sqrt(np.pi*self.Delta))*np.exp(-self.Y**2/(4.0*self.Delta))
		if self.dim==3:
			self.Pn *= (0.5/np.sqrt(np.pi*self.Delta))*np.exp(-self.Z**2/(4.0*self.Delta))
		self.Pn[self.walls_idx] = 0.0
	"""
	@pnt: [x, y, z]
	"""
	def Dirichlet_bc_default(self, t, x, y=None, z=None):
		return 0.0

#######################################
##	Details of numerical methods
#######################################
	"""
	EXPLICITLY define the radial basis function
	"""
	def c_j(self, point_j):
		cmin = -0.05
		cmax =  0.05
		if self.dim==1:
			cj = cmin + (cmax - cmin)*(1+abs(point_j))
		elif self.dim==2:
			cj = cmin + 0.5*(cmax - cmin)*(point_j[0]**2 + point_j[1]**2)**0.5
		elif self.dim==3:
			cj = cmin + (cmax - cmin)*(point_j[0]**2 + point_j[1]**2 + point_j[2])**0.5/3.0
		return cj

	def phi_ij(self, point_i, point_j):
		point_i  = np.array(point_i)
		point_j  = np.array(point_j)
		if self.rbf=="MQ":
			c_j  = self.c_j(point_j)
			phi  = np.sqrt(np.sum((point_i - point_j)**2) + c_j**2)
		return phi

	def D_phi(self, point_i, point_j):
		point_i  = np.array(point_i)
		point_j  = np.array(point_j)	
		if self.rbf=="MQ":
			dphi = []
			c_j  = self.c_j(point_j) 
			if self.dim==1:
				dxPhi = (point_i - point_j)/np.sqrt(np.sum((point_i-point_j)**2) + c_j**2)
			else:
				dxPhi = (point_i[0] - point_j[0])/np.sqrt(np.sum((point_i-point_j)**2) + c_j**2)
			dphi.append(dxPhi)
			if self.dim>=2:
				dyPhi = (point_i[1] - point_j[1])/np.sqrt(np.sum((point_i-point_j)**2) + c_j**2)
				dphi.append(dyPhi)
			if self.dim==3:
				dzPhi = (point_i[2] - point_j[2])/np.sqrt(np.sum((point_i-point_j)**2) + c_j**2)
				dphi.append(dzPhi)
		return np.array(dphi)

	def DD_phi(self, point_i, point_j):
		point_i  = np.array(point_i)
		point_j  = np.array(point_j)	
		phi_ij   = self.phi_ij(point_i, point_j) 
		if self.rbf=="MQ":
			c_j  = self.c_j(point_j)
			if self.dim==1:
				dxdxPhi = (phi_ij**2 - (point_i - point_j)**2)/phi_ij**3
			else:
				dxdxPhi = (phi_ij**2 - (point_i[0] - point_j[0])**2)/phi_ij**3   #PAM: the exponents should be 3 not 1.5
			d2phi = [dxdxPhi]
			if self.dim==2:
				dxdyPhi = -(point_i[0] - point_j[0])*(point_i[1] - point_j[1])/phi_ij**3
				dydxPhi = -(point_i[0] - point_j[0])*(point_i[1] - point_j[1])/phi_ij**3
				dydyPhi =  (phi_ij**2 - (point_i[1] - point_j[1])**2)/phi_ij**3
				d2phi.extend([dxdyPhi, dydxPhi, dydyPhi])
			if self.dim==3:
				dxdyPhi = -(point_i[0] - point_j[0])*(point_i[1] - point_j[1])/phi_ij**3
				dxdzPhi = -(point_i[0] - point_j[0])*(point_i[2] - point_j[2])/phi_ij**3
				dydxPhi = -(point_i[0] - point_j[0])*(point_i[1] - point_j[1])/phi_ij**3
				dydyPhi =  (phi_ij**2 - (point_i[1] - point_j[1])**2)/phi_ij**3
				dydzPhi = -(point_i[1] - point_j[1])*(point_i[2] - point_j[2])/phi_ij**3
				dzdxPhi = -(point_i[2] - point_j[2])*(point_i[0] - point_j[0])/phi_ij**3
				dzdyPhi = -(point_i[2] - point_j[2])*(point_i[1] - point_j[1])/phi_ij**3
				dzdzPhi =  (phi_ij**2 - (point_i[2] - point_j[2])**2)/phi_ij**3
				d2phi.extend([dxdyPhi, dxdzPhi, dydxPhi, dydyPhi, dydzPhi, dzdxPhi, dzdyPhi, dzdzPhi])
		return np.array(d2phi)




	def build_linear_system(self):
		self.tnp1 = self.tn + self.dt
		self.tnim = (self.tn + self.tnp1)/2.0
		# M, G = (NI + NB)*(NI + NB)
		# <- NI -> | <- NB -> 
		self.Phi  = np.zeros((self.N, self.N))
		self.Mnp1 = np.zeros((self.N, self.N))
		self.Gn   = np.zeros((self.N, self.N))
		self.Hnp1 = np.zeros(self.N)
		self.Enim = np.zeros(self.N)
		self.build_boundaries(self.tn, self.tnp1)
		self.M_built = True
		self.build_interiors(self.tn, self.tnp1)
		

	"""
	build the matrix of boundary conditions: 
	update Hnp1 each interation if Dirichlet condition evoles
	no nead to update Mnp1 after initialization
	"""
	def build_boundaries(self, tn, tnp1):
		for i in range(self.NB):
			node_i  = self.walls_idx[i]
			point_i = self.get_point(node_i)
			self.Hnp1[i] = self.Dirichlet_bc(tnp1, point_i)
			if not self.M_built:
				for j in range(self.N):
					if j<self.NI:
						node_j = self.interior_idx[j]
					else:
						node_j = self.walls_idx[j - self.NI]
					point_j = self.get_point(node_j)
					phi_ij  = self.phi_ij(point_i, point_j)
					self.Mnp1[i, j]  = phi_ij
					self.Phi[i, j]   = phi_ij

	def build_interiors(self, tn, tnp1):
		for i in range(self.NI):
			node_i  = self.interior_idx[i]
			point_i = self.get_point(node_i)
			self.Enim[self.NB + i] = self.dt*self.source(self.tnim, point_i) 
			for j in range(self.N):  
				if j<self.NI:
					node_j = self.interior_idx[j]
				else:
					node_j = self.walls_idx[j - self.NI]
				point_j = self.get_point(node_j)
				
				phi     = self.phi_ij(point_i, point_j)
				dPhi_k  = self.D_phi(point_i, point_j)
				d2Phi_k = self.DD_phi(point_i, point_j)
				self.Phi[self.NB + i, j] = phi

				# compute Mnp1
				D_k     = self.drift(self.tnp1, point_i)
				dD_k    = self.D_drift(self.tnp1, point_i)
				D_km    = self.diffusion(self.tnp1, point_i)
				dD_km   = self.D_diffusion(self.tnp1, point_i)
				d2D_km  = self.DD_diffusion(self.tnp1, point_i)
				self.Mnp1[self.NB + i, j] = phi - 0.5*self.dt*(-np.sum(dD_k)*phi - D_k.dot(dPhi_k)) \
									  - 0.5*self.dt*(phi*np.sum(d2D_km) + D_km.flatten().dot(d2Phi_k))
				if self.dim==1:
					self.Mnp1[self.NB + i, j] += -0.5*self.dt*(2*dD_km[0]*dPhi_k[0])
				if self.dim==2:
					self.Mnp1[self.NB + i, j] += -0.5*self.dt*(dD_km[0]*dPhi_k[0] + dD_km[5]*dPhi_k[1] \
													 + dPhi_k[0]*np.sum(dD_km[:3]) + dPhi_k[1]*np.sum(dD_km[3:]))
				if self.dim==3:
					self.Mnp1[self.NB + i, j] += -0.5*self.dt*(dD_km[0]*dPhi_k[0] + dD_km[7]*dPhi_k[1] + dD_km[14]*dPhi_k[2] \
													 + dPhi_k[0]*np.sum(dD_km[:5]) + dPhi_k[1]*np.sum(dD_km[5:10]) + dPhi_k[2]*np.sum(dD_km[10:]))
				# compute Gn
				D_k     = self.drift(self.tn, point_i)
				dD_k    = self.D_drift(self.tn, point_i)
				D_km    = self.diffusion(self.tn, point_i)
				dD_km   = self.D_diffusion(self.tn, point_i)
				d2D_km  = self.DD_diffusion(self.tn, point_i)
				self.Gn[self.NB + i, j] = phi + 0.5*self.dt*(-np.sum(dD_k)*phi - D_k.dot(dPhi_k)) \
									+ 0.5*self.dt*(phi*np.sum(d2D_km) + D_km.flatten().dot(d2Phi_k))
				if self.dim==1:
					self.Gn[self.NB + i, j] += 0.5*self.dt*(2*dD_km[0]*dPhi_k[0])
				if self.dim==2:
					self.Gn[self.NB + i, j] += 0.5*self.dt*(dD_km[0]*dPhi_k[0] + dD_km[5]*dPhi_k[1] \
												  + dPhi_k[0]*np.sum(dD_km[:3]) + dPhi_k[1]*np.sum(dD_km[3:]))
				if self.dim==3:
					self.Gn[self.NB + i, j] += 0.5*self.dt*(dD_km[0]*dPhi_k[0] + dD_km[7]*dPhi_k[1] + dD_km[14]*dPhi_k[2] \
											      + dPhi_k[0]*np.sum(dD_km[:5]) + dPhi_k[1]*np.sum(dD_km[5:10]) + dPhi_k[2]*np.sum(dD_km[10:]))

	def initialize_Lambdas(self):
		if self.QR:
			Q, R = np.linalg.qr(self.Phi)
			y = Q.T.dot(self.Pn)
			self.Lambda_n = np.linalg.solve(R, y)
		else:
			self.Lambda_n = np.linalg.solve(self.Phi, self.Pn)

	def step(self, time_dependent):
		RHS = self.Gn.dot(self.Lambda_n) + self.Hnp1 + self.Enim
		if self.QR:
			if time_dependent:
				self.Q, self.R = np.linalg.qr(self.Mnp1)
			y = self.Q.T.dot(RHS)
			self.Lambda_n = np.linalg.solve(self.R, y)
		else:
			self.Lambda_n = np.linalg.solve(self.Mnp1, RHS)
		self.Pn = self.Phi.dot(self.Lambda_n)

	def solve(self):
		self.initialize_Lambdas()
		plt.ion()
		plt.figure(figsize=(7,7))
		if self.dim==1: plt.plot(self.X, self.Pn)
		time_dependent = True
		self.times = []
		cols = self.get_colors(int(self.tf/self.dt))
		counter = 0
		while self.tn < self.tf:
			if self.tn + self.dt > self.tf:
				self.dt = self.tf - self.tn
			self.times.append(self.tn)
			self.step(time_dependent)
			self.save_statistics()
			if self.dim==1: plt.plot(self.X, self.Pn, color=cols[counter])
			plt.pause(0.001) 
			plt.show()
			self.M_built = False
			time_dependent = self.time_dependent
			self.build_linear_system()
			self.tn += self.dt
			counter += 1
		self.plot_stats()
		

	def save_statistics(self):
		if self.dim==1:
			mass = integrate.simps(self.Pn, self.X) #np.sum(self.Pn)*self.Delta
			mn   = integrate.simps(self.X*self.Pn, self.X) #np.sum(self.X*self.Pn)*self.Delta
			var  = integrate.simps(self.X**2*self.Pn, self.X) - mn**2 #np.sum(self.X**2*self.Pn)*self.Delta - mn**2
			self.x_mean.append(mn)
			self.x_var.append(var)
			self.mass_store.append(mass)
			print('t: ', self.tn, ' mean: ', mn, ' variance: ', var, ' mass: ', mass)

	def plot_stats(self):
		plt.figure(figsize=(7,7))
		plt.plot(self.times, self.x_mean, c='b', label='mean')
		plt.plot(self.times, self.x_var, c='r', label='var')
		plt.plot(self.times, self.mass_store, c='k', label='mass')
		plt.legend(fontsize=20)
		plt.xlabel('time', fontsize=25)
		plt.ylabel('value', fontsize=25)
		plt.tight_layout()
		plt.show()

	def get_colors(self, num, cmmp='coolwarm'):
		cmap = plt.cm.get_cmap(cmmp)
		cs = np.linspace(0, 1, num)
		colors = []
		for i in range(num):
			colors.append(cmap(cs[i]))
		return np.array(colors)


