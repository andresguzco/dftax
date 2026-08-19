"""Tests for utility functions (vmap)."""

import jax
import jax.numpy as jnp

from dftax.utils.vmap import vmap


class TestVmap:

    def test_basic_vmap(self):
        fn = lambda x: x ** 2
        x = jnp.arange(5, dtype=jnp.float32)
        result = vmap(fn)(x)
        expected = jax.vmap(fn)(x)
        assert jnp.allclose(result, expected)

    def test_chunked_vmap(self):
        fn = lambda x: x ** 2
        x = jnp.arange(10, dtype=jnp.float32)
        result_chunked = vmap(fn, chunk_size=3)(x)
        result_full = vmap(fn)(x)
        assert jnp.allclose(result_chunked, result_full)

    def test_in_axes_none(self):
        fn = lambda x, y: x + y
        x = jnp.ones((5, 3))
        y = jnp.array([1.0, 2.0, 3.0])
        result = vmap(fn, in_axes=(0, None))(x, y)
        assert result.shape == (5, 3)
        assert jnp.allclose(result[0], jnp.array([2.0, 3.0, 4.0]))

