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

from typing import Sequence

from jax.config import config
config.update("jax_enable_x64", True)


class Encoder(nn.Module):
    latents: Sequence[int]
    N: int
    n: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.latents[0], dtype=jnp.float64,
                     param_dtype=jnp.float64)(x)
        x = nn.swish(x)
        xr = nn.Dense(self.n, dtype=jnp.float64, param_dtype=jnp.float64)(x)
        return xr


def gaussian_kernel(window_size, sigma):
    mu = window_size / 2
    x = jnp.arange(window_size)
    window = jnp.exp((-((x - mu)**2)) / (2 * sigma**2))
    window = window / jnp.sum(window)
    return window


@jax.jit
def dynamic_gaussian_smooth(xs, windows):
    for i, x in enumerate(xs):
        xs[i] = jnp.convolve(x, windows[i], mode='same')
    return jnp.hstack(xs)


class Decoder(nn.Module):
    latents: Sequence[int]
    N: int
    n: int
    n_sigmas: int

    def setup(self):
        self.split_index = np.linspace(
            0, self.N, self.n_sigmas, endpoint=False, dtype=int)[1:]

    @nn.compact
    def __call__(self, x):
        sigmas = nn.Dense(self.latents[0], dtype=jnp.float64,
                          param_dtype=jnp.float64)(x)
        sigmas = nn.swish(sigmas)
        sigmas = nn.Dense(self.n_sigmas, dtype=jnp.float64,
                          param_dtype=jnp.float64)(sigmas)

        x = nn.Dense(self.N, dtype=jnp.float64, param_dtype=jnp.float64)(x)

        windows = jax.vmap(gaussian_kernel, in_axes=(None, 0))(
            int(self.N / self.n_sigmas), sigmas)
        x = dynamic_gaussian_smooth(jnp.split(x, self.split_index), windows)

        return x


class VAE(nn.Module):
    encoder_latents: Sequence[int]
    decoder_latents: Sequence[int]
    N: int
    n: int
    n_sigmas: int

    def setup(self):
        self.encoder = Encoder(self.encoder_latents, self.N, self.n)
        self.decoder = Decoder(self.decoder_latents,
                               self.N, self.n, self.n_sigmas)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def sigmas(self, x):
        return self.decoder.sigma(x)

    def __call__(self, x):
        return self.decode(self.encode(x))
