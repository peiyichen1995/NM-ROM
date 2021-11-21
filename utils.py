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
    Phi, svals, _ = spla.svd(snapshots, full_matrices=False)

    # Find pod dimension that gives an error below the tolerance
    dim = 1
    err = 1 - np.sum(np.power(svals[:dim], 2)) / np.sum(np.power(svals, 2))
    while (err > TOL and dim < len(svals)):
        dim += 1
        err = 1 - np.sum(np.power(svals[:dim], 2)) / np.sum(np.power(svals, 2))

    return Phi[:, :dim], svals[:dim]


def assemble_reduced_form(form, Phi):
    mat = assemble(form).array()
    red = np.matmul(np.matmul(Phi.T, mat), Phi)
    return red


def solve_svd(A, B):
    # Solve     A X = B
    #    => U S V X = B
    #    =>   S V X = U.T B
    #    =>     V X = 1/S U.T B
    #    =>       X = 1/S V.T U.T B
    U, S, V = spla.svd(A, full_matrices=False)
    SVX = np.matmul(U.T, B)
    VX = np.matmul(np.diag(1 / S), SVX)
    X = np.matmul(V.conj().T, VX)
    return X
