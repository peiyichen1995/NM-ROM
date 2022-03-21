from flax.training import train_state, checkpoints
from flax.core.frozen_dict import FrozenDict
import jax.numpy as jnp
import numpy as np

import jax
from jax import nn as jnn
from jax import random

from functools import partial

from flax import linen as nn
from flax import optim

import optax

import h5py
from fenics import *

from typing import Sequence

import matplotlib.pyplot as plt

from jax.config import config
config.update("jax_enable_x64", True)


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


nu = 0.001
A = 0.5
mesh, u_ref = read_mesh_and_function(
    "output/burgers_1D/nu_" + str(nu) + "/FOM", "u")
u_ref = u_ref.T

time_steps, N = u_ref.shape
u_train = u_ref[np.arange(0, time_steps, 5)]
n_train = len(u_train)
n = 15
M1 = 100
M2 = 100
n_epoch = 40000


class Encoder(nn.Module):
    latents: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.latents[0], dtype=jnp.float64,
                     param_dtype=jnp.float64)(x)
        x = nn.swish(x)
        xr = nn.Dense(n, dtype=jnp.float64, param_dtype=jnp.float64)(x)
        return xr


def gaussian_kernel(window_size, sigma):
    mu = window_size / 2
    x = jnp.arange(window_size)
    window = jnp.exp((-((x - mu)**2)) / (2 * sigma**2))
    window = window / jnp.sum(window)
    return window


@partial(jax.jit, static_argnums=1)
def dynamic_gaussian_smooth(x, window_size, sigmas):
    windows = jax.vmap(gaussian_kernel, in_axes=(None, 0))(window_size, sigmas)
    split_index = np.linspace(0, len(x), len(
        sigmas), endpoint=False, dtype=int)[1:]
    x_split = jnp.split(x, split_index)
    xs = []
    for i, window in enumerate(windows):
        xs.append(jnp.convolve(x_split[i], window, mode='same'))
    return jnp.hstack(xs)


class Decoder(nn.Module):
    latents: Sequence[int]

    @nn.compact
    def __call__(self, x):
        n_sigmas = 5
        sigmas = nn.Dense(n_sigmas, dtype=jnp.float64,
                          param_dtype=jnp.float64)(x)
        x = nn.Dense(self.latents[0], dtype=jnp.float64,
                     param_dtype=jnp.float64)(x)
        x = nn.swish(x)
        x = nn.Dense(N, dtype=jnp.float64, param_dtype=jnp.float64)(x)

        window_size = int(len(x) / 10)
        x = dynamic_gaussian_smooth(x, window_size, sigmas)

        return x


class VAE(nn.Module):
    encoder_latents: Sequence[int]
    decoder_latents: Sequence[int]

    def setup(self):
        self.encoder = Encoder(self.encoder_latents)
        self.decoder = Decoder(self.decoder_latents)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def __call__(self, x):
        return self.decode(self.encode(x))


def model():
    return VAE(encoder_latents=[M1], decoder_latents=[M2])


@jax.jit
def loss_fn(params, x):
    xt = jax.vmap(model().apply, in_axes=(None, 0))(params, x)
    errors = jax.vmap(rel_err, in_axes=(0, 0), out_axes=0)(x, xt)
    l = jnp.sum(errors**2) / n_train
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

for i in range(n_epoch):
    loss_val, grads = loss_grad_fn(params, u_train)
    if loss_val < min_loss:
        min_loss = loss_val
        best_params = best_params.copy(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if i % 10 == 0:
        print('step: {}, loss = {:.6E}, min_loss = {:.6E}'.format(i, loss_val, min_loss))
    if loss_val < 1e-6:
        break

best_loss, _ = loss_grad_fn(best_params, u_train)
print('best loss: {:}'.format(best_loss))

state = train_state.TrainState.create(apply_fn=model().apply,
                                      params=best_params,
                                      tx=tx)
CKPT_DIR = "nu_"+str(nu)
checkpoints.save_checkpoint(
    ckpt_dir=CKPT_DIR, target=state, step=0, overwrite=True)
