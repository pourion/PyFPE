from solver import rbf_fp
import pdb
import numpy as np
import matplotlib.pyplot as plt

dim = 1
Ndim = 60 # use odd number of nodes in each dimension to ensure a node on origin
box = {'xmin':-0.5, 'xmax':1.5, 'ymin':-1, 'ymax':1, 'zmin':-1, 'zmax': 1}
self = rbf_fp(dim, box, Ndim)

# pdb.set_trace()