from fenics import *
import numpy as np

############################
# Model parameters
############################
nu = 1.0

############################
# Mesh
############################
x_left = 0.0
x_right = 2.0
elements = 2000
mesh = IntervalMesh(elements, x_left, x_right)

############################
# Time control
############################
t_start = 0.0
t_final = 0.5
t_steps = 1000
t_sequence = np.linspace(t_start, t_final, t_steps)
dt = (t_final - t_start) / t_steps

############################
# Function space
############################
V = FunctionSpace(mesh, "CG", 1)

############################
# Functions
############################
u = Function(V)
u_old = Function(V)
v = TestFunction(V)

############################
# Boundary conditions
############################
u_left = 1.0
bc_left = DirichletBC(V, u_left, lambda x,
                      on_boundary: on_boundary and near(x[0], x_left))
u_right = 1.0
bc_right = DirichletBC(V, u_right, lambda x,
                       on_boundary: on_boundary and near(x[0], x_right))
bcs = [bc_left, bc_right]

############################
# Initial condition
############################
u0_expr = Expression(
    "x[0] < 1 ? 1+A*(sin(2*pi*x[0]-pi/2)+1) : 1", degree=1, A=nu / 2)
u.interpolate(u0_expr)
u_old.assign(u)

############################
# Weak form and Jacobian
############################
F = (dot(u - u_old, v) / Constant(dt) + inner(u * u.dx(0), v)) * dx
J = derivative(F, u)

############################
# Solve
############################
outfile = XDMFFile("output/burgers_1D.xdmf")
outfile.write(mesh)
outfile.write_checkpoint(u, "u", t_start, XDMFFile.Encoding.HDF5, True)
for t in t_sequence:
    print("----------------------------")
    print("Time = {0:.4E}".format(t))
    problem = NonlinearVariationalProblem(F, u, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.solve()
    u_old.assign(u)
    outfile.write_checkpoint(u, "u", t, XDMFFile.Encoding.HDF5, True)
outfile.close()
