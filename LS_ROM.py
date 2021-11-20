import matplotlib.pyplot as plt
from utils import *
from scipy.integrate import solve_ivp
from numpy.linalg import inv
import math

filename = "output/burgers_1D"

print("Reading mesh and solution of the full order model")
mesh, u = read_mesh_and_function(filename, "u")

print("Performing proper-orthogonal decomposition")
TOL = 1e-6
Phi, svals = POD(u, TOL)
print("{0:d} most important modes selected with a tolerance of {1:.3E}".format(
    len(svals), TOL))

############################
# Model parameters
############################
nu = 1.0

############################
# Function space
############################
V = FunctionSpace(mesh, "CG", 1)

############################
# Functions
############################
u = Function(V)
u_old = Function(V)
u0 = Function(V)
v = TestFunction(V)

############################
# Initial condition
############################
u0_expr = Expression(
    "x[0] < 1 ? 1+A*(sin(2*pi*x[0]-pi/2)+1) : 1", degree=1, A=nu / 2)
u0.interpolate(u0_expr)
u0_red = Phi.T.dot(u0.vector().get_local())

############################
# Weak form and Jacobian
# F = (dot(u - u_old, v) / Constant(dt) + inner(u * u.dx(0), v)) * dx
############################
M_form = derivative(dot(u, v) * dx, u)
M_mat = assemble(M_form).array()
M_red = np.matmul(np.matmul(Phi.T, M_mat), Phi)
M_red_inv = inv(M_red)


def rhs_red(t, u_reduced):
    u.vector().set_local(Phi.dot(u_reduced))
    nl_form = inner(u.dx(0) * u, v) * dx
    nl_red = Phi.T.dot(assemble(nl_form).get_local())
    return -M_red_inv.dot(nl_red)


############################
# Time control
############################
t_start = 0.0
t_final = 0.5
t_steps = 1000
t_sequence = np.linspace(t_start, t_final, t_steps)
dt = (t_final - t_start) / t_steps

############################
# Solve ROM
############################
print("Solving the reduced order model")
u_red = solve_ivp(rhs_red, (t_start, t_final), u0_red,
                  t_eval=t_sequence, method='RK23')

outfile = XDMFFile(filename + "_LS_ROM.xdmf")
i = 0
for t in t_sequence:
    print("Mapping and writing ROM solution at t = {0:.4E}".format(t))
    u.vector().set_local(Phi.dot(u_red.y[:, i]))
    outfile.write(u, t)
    i += 1
outfile.close()
