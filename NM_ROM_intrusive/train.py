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
parser.add_argument('n_sigma', type=int)
args = parser.parse_args()
nu = args.nu
n = args.n
n_sigmas = args.n_sigma

A = 0.5
mesh, u_ref = read_mesh_and_function(
    "../output/burgers_1D/nu_" + str(nu) + "/FOM", "u")
u_ref = u_ref.T

time_steps, N = u_ref.shape
u_train = u_ref[np.arange(0, time_steps, 5)]
n_train = len(u_train)
M1 = 100
M2 = 100
n_epoch = 40000


def model():
    return VAE(encoder_latents=[M1], decoder_latents=[M2], N=N, n=n, n_sigmas=n_sigmas)


@jax.jit
def loss_fn(params, x):
    xt = jax.vmap(model().apply, in_axes=(None, 0))(params, x)
    errors = jax.vmap(rel_err, in_axes=(0, 0), out_axes=0)(x, xt)
    l = jnp.sum(errors**2) / x.shape[0]
    return l


def rel_err(x, xt):
    return jnp.linalg.norm(x - xt)


params = model().init(random.PRNGKey(0), u_train[0])
tx = optax.adam(0.001)
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(loss_fn)
min_loss = loss_fn(params, u_train)
best_params = FrozenDict()
best_params = best_params.copy(params)

CKPT_DIR = "nu_" + str(nu) + "_n_" + str(n) + "_n_sigmas_" + str(n_sigmas)

loss_history = []
for i in range(n_epoch):
    loss_val, grads = loss_grad_fn(params, u_train)
    loss_history.append(loss_val)
    if loss_val < min_loss:
        min_loss = loss_val
        best_params = best_params.copy(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if i % 10 == 0:
        print('step: {}, loss = {:.6E}, min_loss = {:.6E}'.format(
            i, loss_val, min_loss))
    if i % 100 == 0:
        state = train_state.TrainState.create(apply_fn=model().apply,
                                              params=params,
                                              tx=tx)
        checkpoints.save_checkpoint(
            ckpt_dir=CKPT_DIR, target=state, step=i, overwrite=True)
    if loss_val < 1e-6:
        break

state = train_state.TrainState.create(apply_fn=model().apply,
                                      params=best_params,
                                      tx=tx)
checkpoints.save_checkpoint(
    ckpt_dir=CKPT_DIR, target=state, step=n_epoch, overwrite=True)

with open(CKPT_DIR + '/loss.csv', 'w') as f:
    for loss in loss_history:
        f.write("{:.6E}\n".format(loss))
