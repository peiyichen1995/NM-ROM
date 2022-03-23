from model_definition import *
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import FrozenDict

import numpy as np

import h5py
from fenics import *

import matplotlib.pyplot as plt

import argparse


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


parser = argparse.ArgumentParser()
parser.add_argument('nu', type=float)
parser.add_argument('n', type=int)
args = parser.parse_args()
nu = args.nu
n = args.n

A = 0.5
mesh, u_ref = read_mesh_and_function(
    "../output/burgers_1D/nu_" + str(nu) + "/FOM", "u")
u_ref = u_ref.T

time_steps, N = u_ref.shape
u_train = np.copy(u_ref)
n_train = len(u_train)
M1 = 100


def model():
    return VAE(encoder_latents=[M1], N=N, n=n)


@jax.jit
def loss_fn(params, x):
    xt = jax.vmap(model().apply, in_axes=(None, 0))(params, x)
    errors = jax.vmap(rel_err, in_axes=(0, 0), out_axes=0)(x, xt)
    l = jnp.sum(errors**2) / x.shape[0]
    return l


def rel_err(x, xt):
    return jnp.linalg.norm(x - xt)


n_epoch = 40000
n_batches = 25
learning_rate = 0.001
learning_rate_cut_factor = 10
max_patience = 200
training_tol = 1e-4

params = model().init(random.PRNGKey(0), u_train[46])
tx = optax.adam(learning_rate)
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(loss_fn)
min_loss = 1e6
best_params = FrozenDict()
best_params = best_params.copy(params)
batch_indices = np.linspace(0, u_train.shape[0], n_batches, endpoint=False)[1:]
CKPT_DIR = "nu_" + str(nu) + "_n_" + str(n)

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
    # if i % 10 == 0:
    print('Epoch {}, loss = {:.6E}, min_loss = {:.6E}'.format(
        i, loss_val, min_loss))
    if i % 100 == 0:
        state = train_state.TrainState.create(apply_fn=model().apply,
                                              params=params,
                                              tx=tx)
        checkpoints.save_checkpoint(
            ckpt_dir=CKPT_DIR, target=state, step=i, overwrite=True)
    if loss_val < training_tol:
        break

state = train_state.TrainState.create(apply_fn=model().apply,
                                      params=best_params,
                                      tx=tx)
checkpoints.save_checkpoint(
    ckpt_dir=CKPT_DIR, target=state, step=n_epoch, overwrite=True)

with open(CKPT_DIR + '/loss.csv', 'w') as f:
    for loss in loss_history:
        f.write("{:.6E}\n".format(loss))
