import matplotlib
import matplotlib.pyplot as plt
from utils import *
from scipy.integrate import solve_ivp
from numpy.linalg import inv

############################
# Model parameters
############################
method = "LS_ROM_DMD_fully_offline"
nu = 0.001
A = 0.5
filename = "output/burgers_1D/nu_" + str(nu)

############################
# Read snapshots
############################
print("Reading mesh and solution of the full order model")
mesh, u_ref = read_mesh_and_function(filename, "u")
mesh, u_ref_dot = read_mesh_and_function(filename, "u_dot")

############################
# Perform POD
############################
print("Performing proper-orthogonal decomposition")
TOL = 1e-12
Phi, svals = POD(u_ref, TOL)
print("{0:d} most important modes selected with a tolerance of {1:.3E}".format(
    len(svals), TOL))

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
    "x[0] < 1 ? 1+A*(sin(2*pi*x[0]-pi/2)+1) : 1", degree=1, A=A)
u0.interpolate(u0_expr)
u0_red = Phi.T.dot(u0.vector().get_local())

############################
# Dynamic mode decomposition
# Approximate the linear operator A:
# du/dt = A u
############################
u_ref_red = np.matmul(Phi.T, u_ref)
u_ref_dot_red = np.matmul(Phi.T, u_ref_dot)
A_red = solve_svd(u_ref_red.T, u_ref_dot_red.T)
A_red = A_red.T


def rhs_red(t, u_red):
    return A_red.dot(u_red)


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

outfile = XDMFFile(filename + "_" + method + ".xdmf")
i = 0
u_proj = np.zeros(u_ref.shape)
u_proj[:, 0] = Phi.dot(u0_red)
for t in t_sequence:
    print("Mapping and writing ROM solution at t = {0:.4E}".format(t))
    u_proj[:, i + 1] = Phi.dot(u_red.y[:, i])
    u.vector().set_local(u_proj[:, i + 1])
    outfile.write(u, t)
    i += 1
outfile.close()

############################
# Plot results
############################
plt.rcParams.update({'font.size': 16})

# Reference solution
fig, ax = plt.subplots()
im = ax.imshow(u_ref)
cb_ref = fig.colorbar(im)
ax.set_xlabel("$t$")
ax.set_ylabel("$x$")
ax.set_xticks(np.linspace(0, t_steps, 3))
ax.set_xticklabels(np.linspace(t_start, t_final, 3))
ytick_loc = np.linspace(0, V.dim() - 1, 3).astype(int)
ax.set_yticks(ytick_loc)
ax.set_yticklabels(V.tabulate_dof_coordinates()[ytick_loc, 0])
ax.set_aspect("auto")
plt.tight_layout(pad=0)
plt.savefig(filename + "_ref.png")
plt.close()

# Projected ROM solution
fig, ax = plt.subplots()
im = ax.imshow(u_proj, vmin=1, vmax=2)
cb = fig.colorbar(im)
cb.set_ticks(cb_ref.get_ticks())
ax.set_xlabel("$t$")
ax.set_ylabel("$x$")
ax.set_xticks(np.linspace(0, t_steps, 3))
ax.set_xticklabels(np.linspace(t_start, t_final, 3))
ytick_loc = np.linspace(0, V.dim() - 1, 3).astype(int)
ax.set_yticks(ytick_loc)
ax.set_yticklabels(V.tabulate_dof_coordinates()[ytick_loc, 0])
ax.set_aspect("auto")
plt.tight_layout(pad=0)
plt.savefig(filename + "_" + method + "_proj.png")
plt.close()

# Error
fig, ax = plt.subplots()
im = ax.imshow(np.abs(u_proj - u_ref), norm=matplotlib.colors.LogNorm())
fig.colorbar(im)
ax.set_xlabel("$t$")
ax.set_ylabel("$x$")
ax.set_xticks(np.linspace(0, t_steps, 3))
ax.set_xticklabels(np.linspace(t_start, t_final, 3))
ytick_loc = np.linspace(0, V.dim() - 1, 3).astype(int)
ax.set_yticks(ytick_loc)
ax.set_yticklabels(V.tabulate_dof_coordinates()[ytick_loc, 0])
ax.set_aspect("auto")
plt.tight_layout(pad=0)
plt.savefig(filename + "_" + method + "_error.png")
plt.close()
