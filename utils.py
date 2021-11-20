import h5py
from fenics import *
import numpy as np
import scipy.linalg as spla


def read_mesh_and_function(file_name, var_name):

    # Open solution file
    infile = XDMFFile(file_name + ".xdmf")
    infile_h5 = h5py.File(file_name + ".h5", "r")
    t_steps = len(infile_h5[var_name].keys())

    # Read in mesh
    mesh = Mesh()
    infile.read(mesh)

    # Read function
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    solution = np.zeros((V.dim(), t_steps))
    for i in range(t_steps):
        infile.read_checkpoint(u, var_name, i - t_steps + 1)
        solution[:, i] = u.vector().get_local()

    # Clean up
    infile.close()
    infile_h5.close()

    return mesh, solution


def POD(snapshots, TOL=0):
    phi, svals, _ = spla.svd(snapshots, full_matrices=False)

    # Find pod dimension that gives an error below the tolerance
    dim = 1
    err = 1 - np.sum(svals[:dim]) / np.sum(svals)
    while (err > TOL and dim < len(svals)):
        dim += 1
        err = 1 - np.sum(svals[:dim]) / np.sum(svals)

    return phi[:, :dim], svals[:dim]
