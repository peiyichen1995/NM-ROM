import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class NonlinearReducedBasisSurrogate(nn.Module):
    encoder_latents: Sequence[int]
    N: int
    n: int

    def setup(self):
        self.encoder = self.Encoder(self.encoder_latents, self.n)
        self.decoder = self.Decoder(self.N, self.n)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def __call__(self, x):
        return self.decode(self.encode(x))

    class Encoder(nn.Module):
        latents: Sequence[int]
        n: int

        @nn.compact
        def __call__(self, x):
            for latent in self.latents:
                x = nn.Dense(latent, dtype=jnp.float64,
                             param_dtype=jnp.float64)(x)
                x = nn.swish(x)
            x = nn.Dense(self.n, dtype=jnp.float64,
                         param_dtype=jnp.float64)(x)
            return x

    class Decoder(nn.Module):
        N: int
        n: int

        def bubble(self, window_size, s):
            mu = window_size / 2
            x = jnp.arange(window_size)
            window = nn.relu(- (x - mu)**2 /
                             (0.5 * s * window_size)**2 + 1)
            window = window / jnp.sum(window)
            return window

        @nn.compact
        def __call__(self, x):
            sub_weights = self.param(
                'sub_weight', lambda key: jnp.ones((self.n,)))
            sub_bias = self.param('sub_bias', lambda key: jnp.zeros((self.N,)))

            sub_sigmas = nn.Dense(
                self.N, name='sub_sigma_1', dtype=jnp.float64, param_dtype=jnp.float64)(x)
            sub_sigmas = nn.sigmoid(sub_sigmas)
            sub_sigmas = nn.Dense(
                self.n, name='sub_sigma_2', dtype=jnp.float64, param_dtype=jnp.float64)(sub_sigmas)
            sub_sigmas = nn.sigmoid(sub_sigmas)

            sub_windows = jax.vmap(self.bubble, in_axes=(
                None, 0))(self.N / 20, sub_sigmas)
            x_net = jnp.zeros((self.N,))
            for i in range(self.n):
                sub_x = nn.Dense(self.N, dtype=jnp.float64,
                                 param_dtype=jnp.float64)([x[i]])
                x_net = x_net + sub_weights[i] * \
                    jnp.convolve(sub_x, sub_windows[i], mode='same')

            return x_net + sub_bias
