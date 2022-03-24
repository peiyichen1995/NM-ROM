import matplotlib
import matplotlib.pyplot as plt
from utils import *
from scipy.integrate import solve_ivp
from numpy.linalg import inv

############################
# Model parameters
############################
method = "NM_ROM"
nu = 0.001
A = 0.5
filename = "output/burgers_1D/nu_" + str(nu) + "/"

############################
# Read snapshots
############################
print("Reading mesh and solution of the full order model")
mesh, u_ref = read_mesh_and_function(filename + "FOM", "u")

############################
# Perform AutoEncoding
# R: number of dofs in the full order model
# r: number of dofs in the reduced order model
############################

# This is the number of dofs in the reduced order model
r = 82

R, time_steps = u_ref.shape

# split test and train data
X_train, X_test, y_train, y_test = train_test_split(
    u_ref.T, u_ref.T, test_size=0.2, random_state=42)

# This is our input image
input_img = keras.Input(shape=(R, ))

# "encoded" is the encoded representation of the input
# encoded = layers.Dense(r, activation='sigmoid')(input_img)
encoded = layers.Dense(1500, activation='linear')(input_img)
encoded = layers.Dense(r, activation='linear')(encoded)


# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(R, activation='linear')(encoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

# This model maps an input to its encoded representation
encoder = keras.Model(input_img, encoded)

# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(r,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

num_epochs = 1

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(X_train, X_train,
                epochs=num_epochs,
                validation_data=(X_test, X_test))


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

print(input_img.shape)

u0_red = encoder.predict(u0.vector().get_local().reshape(1, -1))
u0_red = u0_red.reshape(r,)


############################
# Weak form and Jacobian
# F = (dot(u - u_old, v) / Constant(dt) +
#     inner(nu * grad(u), grad(v)) + inner(u * u.dx(0), v)) * dx
############################
M_form = derivative(dot(u, v) * dx, u)
M_red = assemble_reduced_form(M_form, Phi)
M_red_inv = inv(M_red)

K_form = derivative(inner(nu * grad(u), grad(v)) * dx, u)
K_red = assemble_reduced_form(K_form, Phi)


def rhs_red(t, u_red):
    u.vector().set_local(Phi.dot(u_red))
    nl_form = inner(u.dx(0) * u, v) * dx
    nl_red = Phi.T.dot(assemble(nl_form).get_local())
    return -M_red_inv.dot(K_red.dot(u_red) + nl_red)


# ############################
# # Time control
# ############################
# t_start = 0.0
# t_final = 0.5
# t_steps = 500
# t_sequence = np.linspace(t_start, t_final, t_steps + 1)
# dt = (t_final - t_start) / t_steps
#
# ############################
# # Solve ROM
# ############################
# print("Solving the reduced order model")
# u_red = solve_ivp(rhs_red, (t_start, t_final), u0_red,
#                   t_eval=t_sequence, method='RK23')
#
# outfile = XDMFFile(filename + method + ".xdmf")
# u_proj = np.zeros((V.dim(), u_red.y.shape[1]))
# for i in range(u_red.y.shape[1]):
#     print("Mapping and writing ROM solution at t = {0:.4E}".format(
#         t_sequence[i]))
#     u_proj[:, i] = Phi.dot(u_red.y[:, i])
#     u.vector().set_local(u_proj[:, i])
#     outfile.write(u, t_sequence[i])
# outfile.close()
#
#
# ############################
# # Plot results
# ############################
# plt.jet()
# plt.rcParams.update({'font.size': 16})
#
# # Reference solution
# fig, ax = plt.subplots()
# im = ax.imshow(u_ref)
# cb_ref = fig.colorbar(im)
# ax.set_xlabel("$t$")
# ax.set_ylabel("$x$")
# ax.set_xticks(np.linspace(0, t_steps, 3))
# ax.set_xticklabels(np.linspace(t_start, t_final, 3))
# ytick_loc = np.linspace(0, V.dim() - 1, 3).astype(int)
# ax.set_yticks(ytick_loc)
# ax.set_yticklabels(V.tabulate_dof_coordinates()[ytick_loc, 0])
# ax.set_aspect("auto")
# plt.tight_layout(pad=0)
# plt.savefig(filename + "ref.png")
# plt.close()
#
# # Projected ROM solution
# fig, ax = plt.subplots()
# im = ax.imshow(u_proj)
# cb = fig.colorbar(im)
# cb.set_ticks(cb_ref.get_ticks())
# ax.set_xlabel("$t$")
# ax.set_ylabel("$x$")
# ax.set_xticks(np.linspace(0, t_steps, 3))
# ax.set_xticklabels(np.linspace(t_start, t_final, 3))
# ytick_loc = np.linspace(0, V.dim() - 1, 3).astype(int)
# ax.set_yticks(ytick_loc)
# ax.set_yticklabels(V.tabulate_dof_coordinates()[ytick_loc, 0])
# ax.set_aspect("auto")
# plt.tight_layout(pad=0)
# plt.savefig(filename + method + "_sol.png")
# plt.close()
#
# # Error
# fig, ax = plt.subplots()
# im = ax.imshow(np.abs(u_proj - u_ref), norm=matplotlib.colors.LogNorm())
# fig.colorbar(im)
# ax.set_xlabel("$t$")
# ax.set_ylabel("$x$")
# ax.set_xticks(np.linspace(0, t_steps, 3))
# ax.set_xticklabels(np.linspace(t_start, t_final, 3))
# ytick_loc = np.linspace(0, V.dim() - 1, 3).astype(int)
# ax.set_yticks(ytick_loc)
# ax.set_yticklabels(V.tabulate_dof_coordinates()[ytick_loc, 0])
# ax.set_aspect("auto")
# plt.tight_layout(pad=0)
# plt.savefig(filename + method + "_err.png")
# plt.close()
