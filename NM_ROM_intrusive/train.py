
import argparse

import numpy as np

# machine learning packages
import jax  # nopep8
import jax.numpy as jnp  # nopep8
jax.config.update("jax_enable_x64", True)  # nopep8
from flax.core.frozen_dict import FrozenDict  # nopep8
from flax.training import train_state, checkpoints  # nopep8
import optax  # nopep8

# our surrogates and FEM stuff
import sys  # nopep8
sys.path.append('..')  # nopep8
from fenics import XDMFFile, Mesh  # nopep8
from fenics import FunctionSpace, Function  # nopep8
from surrogates import NonlinearReducedBasisSurrogate as NRBS  # nopep8


# Read parameters from cli args
parser = argparse.ArgumentParser()
parser.add_argument('nu', type=float)
parser.add_argument('n', type=int)
args = parser.parse_args()

# Read mesh and solution
nu = args.nu
A = 0.5
t_start = 0.0
t_final = 0.5
t_steps = 501
ts = np.linspace(t_start, t_final, t_steps)

file_name = "../output/burgers_1D/nu_" + str(nu) + "/FOM"
xdmffile = XDMFFile(file_name + ".xdmf")

mesh = Mesh()
xdmffile.read(mesh)
V = FunctionSpace(mesh, 'CG', 1)
X = V.tabulate_dof_coordinates()[:, 0]
u = Function(V)
N = V.dim()

u_ref = np.zeros((t_steps, N))
for i in range(t_steps):
    xdmffile.read_checkpoint(u, 'u', i)
    u_ref[i] = u.vector().get_local()

xdmffile.close()

# Training data and surrogate
u_train = np.copy(u_ref)
n_train = len(u_train)

n = args.n
M1 = 100


def surrogate():
    return NRBS(encoder_latents=[M1], N=N, n=n)


# Loss function
@jax.jit
def loss_fn(params, x):
    xt = jax.vmap(surrogate().apply, in_axes=(None, 0))(params, x)
    errors = jax.vmap(rel_err, in_axes=(0, 0), out_axes=0)(x, xt)
    l = jnp.sum(errors**2) / x.shape[0]
    return l


def rel_err(x, xt):
    return jnp.linalg.norm(x - xt)


# Training parameters
n_epoch = 10000
n_batches = 25
learning_rate = 0.001
learning_rate_cut_factor = 10
max_patience = 200
training_tol = 1e-4

params = surrogate().init(jax.random.PRNGKey(0), u_train[0])
tx = optax.adam(learning_rate)
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(loss_fn)
min_loss = 1e6
best_params = FrozenDict()
best_params = best_params.copy(params)
batch_indices = np.linspace(0, u_train.shape[0], n_batches, endpoint=False)[1:]
CKPT_DIR = "nu_" + str(nu) + "_n_" + str(n)


# Start training
loss_history = []
patience = 0
for i in range(n_epoch):
    if patience > max_patience:
        if learning_rate < 1e-12:
            print('The learning rate is too small. Let us stop here.')
            break
        learning_rate = learning_rate / learning_rate_cut_factor
        print('Min loss has not dropped in {:} epochs. Reduce learning rate to {:}'.format(
            max_patience, learning_rate))
        tx = optax.adam(learning_rate)
        opt_state = tx.init(params)
        patience = 0
    np.random.shuffle(u_train)
    batches = jnp.split(u_train, batch_indices)
    losses = []
    for j, batch in enumerate(batches):
        loss_val, grads = loss_grad_fn(params, batch)
        losses.append(loss_val)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    loss_val = np.mean(losses)
    if loss_val < min_loss:
        min_loss = loss_val
        best_params = best_params.copy(params)
        patience = 0
    else:
        patience = patience + 1
    loss_history.append(loss_val)
    print('Epoch {}, loss = {:.6E}, min_loss = {:.6E}'.format(
        i, loss_val, min_loss))
    if i % 100 == 0:
        state = train_state.TrainState.create(apply_fn=surrogate().apply,
                                              params=params,
                                              tx=tx)
        checkpoints.save_checkpoint(
            ckpt_dir=CKPT_DIR, target=state, step=i, overwrite=True)
    if loss_val < training_tol:
        break

# Save the final training state
state = train_state.TrainState.create(apply_fn=surrogate().apply,
                                      params=best_params,
                                      tx=tx)
checkpoints.save_checkpoint(
    ckpt_dir=CKPT_DIR, target=state, step=n_epoch, overwrite=True)

with open(CKPT_DIR + '/loss.csv', 'w') as f:
    for loss in loss_history:
        f.write("{:.6E}\n".format(loss))
