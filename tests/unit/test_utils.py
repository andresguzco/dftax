"""Tests for utility functions (vmap, energy_aux)."""

import jax
import jax.numpy as jnp

from dftax.utils.vmap import vmap
from dftax.utils.energy_aux import EnergyAux, pack_energy_aux


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


class TestEnergyAux:

    def test_pack_unpack(self):
        aux = pack_energy_aux(
            kinetic=1.0, hartree=2.0, xc=3.0, external=4.0, nelec=10.0,
        )
        assert aux.kinetic == 1.0
        assert aux.hartree == 2.0
        assert aux.xc == 3.0
        assert aux.external == 4.0
        assert aux.nelec == 10.0

    def test_named_tuple_fields(self):
        fields = EnergyAux._fields
        assert "kinetic" in fields
        assert "hartree" in fields
        assert "xc" in fields
        assert "external" in fields
        assert "nelec" in fields

    def test_optional_fields_default_none(self):
        aux = pack_energy_aux(
            kinetic=1.0, hartree=2.0, xc=3.0, external=4.0, nelec=10.0,
        )
        assert aux.kl_div is None
        assert aux.tv_dist is None
        assert aux.entropy is None
