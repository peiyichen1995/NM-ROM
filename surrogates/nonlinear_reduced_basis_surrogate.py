import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class NonlinearReducedBasisSurrogate(nn.Module):
    encoder_latents: Sequence[int]
    decoder_latents: Sequence[int]
    N: int
    n: int
    mu: jnp.float64

    def setup(self):
        self.encoder = self.Encoder(self.encoder_latents, self.n)
        self.decoder = self.Decoder(
            self.decoder_latents, self.N, self.n, self.mu)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def smoothness(self, x):
        return self.decoder.smoothness_map(self.encode(x))

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
        latents: Sequence[int]
        N: int
        n: int
        mu: jnp.float64

        def setup(self):
            self.smoothness_map = self.SmoothnessMap(self.latents, self.n)

        def bubble(self, w):
            x = jnp.arange(2 * self.mu)
            window = nn.relu(- (x - self.mu)**2 / (w * self.mu)**2 + 1)
            window = window / jnp.sum(window)
            return window

        @nn.compact
        def __call__(self, x):
            w = self.smoothness_map(x)
            sub_windows = jax.vmap(self.bubble)(w)

            x_net = jnp.zeros((self.N,))
            for i in range(self.n):
                sub_x = nn.Dense(self.N, dtype=jnp.float64,
                                 param_dtype=jnp.float64)([x[i]])
                x_net = x_net + \
                    jnp.convolve(sub_x, sub_windows[i], mode='same')

            return x_net

        class SmoothnessMap(nn.Module):
            latents: Sequence[int]
            n: int

            @nn.compact
            def __call__(self, x):
                for latent in self.latents:
                    x = nn.Dense(
                        latent, dtype=jnp.float64, param_dtype=jnp.float64)(x)
                    x = nn.sigmoid(x)
                x = nn.Dense(
                    self.n, dtype=jnp.float64, param_dtype=jnp.float64)(x)
                x = nn.sigmoid(x)
                return x
