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

import matplotlib
import matplotlib.pyplot as plt

import basix
from basix import ElementFamily, CellType, LagrangeVariant

import os

from matplotlib.animation import FuncAnimation

plt.rcParams['text.usetex'] = True

# again, this only works on startup!
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
    "../output/burgers_1D/nu_" + str(nu) + "/FOM", "u")
u_ref = u_ref.T
V = FunctionSpace(mesh, "CG", 1)

time_steps, N = u_ref.shape
n = 10
M1 = 100
M2 = 100
n_sigmas = 5

from model_definition import *

def model():
    return VAE(encoder_latents=[M1], decoder_latents=[M2], N=N, n=n, n_sigmas=n_sigmas)

params = model().init(random.PRNGKey(0), u_ref[0])
tx = optax.adam(0.001)
state = train_state.TrainState.create(apply_fn=model().apply,
                                      params=params,
                                      tx=tx)
CKPT_DIR = "nu_" + str(nu) + "_n_" + str(n) + "_n_sigmas_" + str(n_sigmas)
restored_state = checkpoints.restore_checkpoint(
    ckpt_dir=CKPT_DIR, target=state)
params = params.copy(restored_state.params)

from matplotlib.animation import FuncAnimation

fig = plt.figure(figsize=(6, 8))
(ax1, ax2) = fig.subplots(2, 1, sharex=True)
lines1 = ax1.plot([])
ax1.set_xlim(0, 2)
ax1.set_ylim(0.5, 2.5)
ax1.set_ylabel('u')
lines2 = ax2.plot(np.empty((0, 3)), np.empty((0, 3)))
ax2.set_xlim(0, 2)
ax2.set_ylim(-0.3, 0.3)
ax2.set_ylabel('\phi')
ax2.set_xlabel('x')

Phi_func = jax.jit(jax.jacfwd(lambda params, u: model().apply(params, u, method=VAE.decode), argnums=1))

def AnimationFunction(frame):
    if frame % 100 == 0:
        print('frame {:}'.format(frame))
    u_ref_encoded = model().apply(params, u_ref[frame], method=VAE.encode)
    Phi = Phi_func(params, u_ref_encoded)
    lines1[0].set_data((V.tabulate_dof_coordinates()[:,0], u_ref[frame]))
    lines2[0].set_data((V.tabulate_dof_coordinates()[:,0], Phi.T[0]))
    lines2[1].set_data((V.tabulate_dof_coordinates()[:,0], Phi.T[1]))
    lines2[2].set_data((V.tabulate_dof_coordinates()[:,0], Phi.T[2]))

anim = FuncAnimation(fig, AnimationFunction, frames=u_ref.shape[0], interval=15)
video = anim.to_html5_video()
with open("figures/nu_" + str(nu) + "_basis_n_" + str(n) + ".html", "w") as f:
    f.write(video)

plt.close()

plt.figure(figsize=(20, 6), dpi=80)
u_peek = u_ref[0]
plt.plot(u_peek)
plt.plot(model().apply(params, u_peek))