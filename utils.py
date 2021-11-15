from dolfin import *
import meshio
import numpy as np
import math
import scipy.linalg as dla
from scipy.stats import gamma
from scipy.stats import norm


def build_nullspace(V):
    x = Function(V).vector()
    nullspace_basis = [x.copy() for i in range(6)]
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0)
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0)
    V.sub(2).dofmap().set(nullspace_basis[2], 1.0)
    V.sub(0).set_x(nullspace_basis[3], -1.0, 1)
    V.sub(1).set_x(nullspace_basis[3],  1.0, 0)
    V.sub(0).set_x(nullspace_basis[4],  1.0, 2)
    V.sub(2).set_x(nullspace_basis[4], -1.0, 0)
    V.sub(1).set_x(nullspace_basis[5], -1.0, 2)
    V.sub(2).set_x(nullspace_basis[5],  1.0, 1)
    for x in nullspace_basis:
        x.apply("insert")
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()
    return basis


def build_nullspace_three_field(V, i):
    x = Function(V).vector()
    nullspace_basis = [x.copy() for i in range(6)]
    V.sub(i).sub(0).dofmap().set(nullspace_basis[0], 1.0)
    V.sub(i).sub(1).dofmap().set(nullspace_basis[1], 1.0)
    V.sub(i).sub(2).dofmap().set(nullspace_basis[2], 1.0)
    V.sub(i).sub(0).set_x(nullspace_basis[3], -1.0, 1)
    V.sub(i).sub(1).set_x(nullspace_basis[3],  1.0, 0)
    V.sub(i).sub(0).set_x(nullspace_basis[4],  1.0, 2)
    V.sub(i).sub(2).set_x(nullspace_basis[4], -1.0, 0)
    V.sub(i).sub(1).set_x(nullspace_basis[5], -1.0, 2)
    V.sub(i).sub(2).set_x(nullspace_basis[5],  1.0, 1)
    for x in nullspace_basis:
        x.apply("insert")
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()
    return basis


def matrix_cofactor(matrix):
    return det(matrix) * inv(matrix)


def my_cross(a, b):
    return as_vector((a[1] * b[2] - a[2] * b[1],  a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))


def MSH2XDMF(MSH_name, XDMF_mesh_name, XDMF_mesh_face_name):
    msh = meshio.read(MSH_name)
    for cell in msh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
        elif cell.type == "tetra":
            tetra_cells = cell.data

    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if key == "triangle":
            triangle_data = msh.cell_data_dict["gmsh:physical"][key]
        elif key == "tetra":
            tetra_data = msh.cell_data_dict["gmsh:physical"][key]

    tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells})
    triangle_mesh = meshio.Mesh(points=msh.points,
                                cells=[("triangle", triangle_cells)],
                                cell_data={"face_id": [triangle_data]})

    meshio.write(XDMF_mesh_name, tetra_mesh)
    meshio.write(XDMF_mesh_face_name, triangle_mesh)


def XDMF2PVD(XDMF_mesh_name, XDMF_mesh_face_name, PVD_mesh_name, PVD_mesh_face_name):
    mesh = Mesh()
    with XDMFFile(XDMF_mesh_name) as infile:
        infile.read(mesh)
    File(PVD_mesh_name).write(mesh)

    mvc = MeshValueCollection("size_t", mesh, 1)
    with XDMFFile(XDMF_mesh_face_name) as infile:
        infile.read(mvc, "face_id")
    mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    File(PVD_mesh_face_name).write(mf)

    return mesh, mf


def shortest_dis(radius, points, point, threshold1, threshold2):
    distance = point.distance(points[0])
    id = 0
    for i in range(len(points)):
        temp = point.distance(points[i])
        if(temp < distance):
            distance = temp
            id = i
    return (radius[id] * (1 + threshold1)) < distance < (radius[id] * (1 + threshold2))

# exponential covariance function


def cov_exp(r, rho, sigma=1.0):
    return sigma * np.exp(-math.pi * r * r / 2.0 / rho / rho)

# covariance length


def cov_len(rho, sigma=1.0):
    return integrate.quad(lambda r: cov_exp(r, rho), 0, math.inf)


def set_fem_fun(vec, fs):
    retval = Function(fs)
    retval.vector().set_local(vec)
    return retval

# Find order for truncation error that is smaller than err


def trun_order(err, C, M, w):
    e = 0
    eig = 0
    trCM = np.trace(np.dot(C, M))
    while 1 - eig / trCM > 0.1:
        eig = eig + w[e]
        e = e + 1
    error = (1 - eig / trCM)
    return e, error


def solve_covariance_EVP(cov, mesh, V):
    u = TrialFunction(V)
    v = TestFunction(V)

    # dof to vertex map
    dof2vert = dof_to_vertex_map(V)
    # coords will be used for interpolation of covariance kernel
    coords = mesh.coordinates()
    # but we need degree of freedom ordering of coordinates
    coords = coords[dof2vert]

    # assemble mass matrix and convert to scipy
    M = assemble(u * v * dx)
    M = M.array()

    # evaluate covariance matrix
    L = coords.shape[0]
    C = np.zeros([L, L])

    for i in range(L):
        for j in range(L):
            if j <= i:
                v = cov(np.linalg.norm(coords[i] - coords[j]))
                C[i, j] = v
                C[j, i] = v

    # solve eigenvalue problem
    A = np.dot(M, np.dot(C, M))

    # w, v = spla.eigsh(A, k, M)
    w, v = dla.eigh(A, b=M)

    return w, v, C, M


# order eigenvalues and eigen Vectors
def order_eig(w, v):
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:, idx]
    return w, v

# generate non-gaussian randomField


def nonGauss(w, v, loc, scale, e):
    randomField = np.zeros(v[:, 0].shape)
    gauss = np.random.normal(loc=0.0, scale=1.0, size=(len(w), 1))
    for i in range(e):
        randomField = randomField + sqrt(w[i]) * v[:, i] * gauss[i]
    for i in range(len(w)):
        randomField[i] = norm.cdf(randomField[i])
        randomField[i] = gamma.ppf(
            randomField[i], 1 / scale, loc=loc, scale=scale)
    return randomField
