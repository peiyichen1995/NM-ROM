import keras
from keras import layers
import h5py
from fenics import *
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split

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
    Phi, svals, _ = sp.linalg.svd(snapshots, full_matrices=False)

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


def solve_svd(A, B, TOL=1e-6):
    # Solve       A X = B
    #    => U S V.T X = B
    #    =>   S V.T X = U.T B
    #    =>     V.T X = 1/S U.T B
    #    =>         X = 1/S V U.T B
    U, S, Vt = sp.linalg.svd(A, full_matrices=False)

    # Eliminate zero singular values
    dim = 1
    err = 1 - np.sum(np.power(S[:dim], 2)) / np.sum(np.power(S, 2))
    while (err > TOL and dim <= len(S)):
        dim += 1
        err = 1 - np.sum(np.power(S[:dim], 2)) / np.sum(np.power(S, 2))
    U = U[:, :dim]
    S = S[:dim]
    Vt = Vt[:dim, :]

    SVtX = np.matmul(U.T, B)
    VtX = np.matmul(np.diag(1 / S), SVtX)
    X = np.matmul(Vt.conj().T, VtX)
    return X

# def encoder_decoder(snapshots, TOL=0):
