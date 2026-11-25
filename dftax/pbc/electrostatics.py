from __future__ import annotations

import jax
import jax.numpy as jnp

from dftax.pbc.gth import vloc_G
from dftax.pbc.lattice import gvectors, volume


def coulomb_kernel(a, mesh):
    """``4π / G^2`` on the plane waves of a mesh (zero at ``G = 0``), ``(N,)``."""
    G = gvectors(a, mesh)
    G2 = jnp.sum(G * G, axis=-1)
    return jnp.where(G2 > 0.0, 4.0 * jnp.pi / jnp.where(G2 > 0.0, G2, 1.0), 0.0)


def hartree_energy(rho, a, mesh):
    """Hartree energy of the density ``rho`` (``(N,)``, grid order) of a cell
    with lattice rows ``a``."""
    mesh = tuple(int(n) for n in mesh)
    rho_G = jnp.fft.fftn(rho.reshape(mesh)).reshape(-1)
    n = rho.shape[0]
    return 0.5 * volume(a) / n ** 2 * jnp.sum(
        coulomb_kernel(a, mesh) * (rho_G.real ** 2 + rho_G.imag ** 2))


def local_potential(coords, a, mesh, zion, rloc, cloc):
    """Local pseudopotential of all atoms on the uniform grid, already times
    the quadrature weight: ``(Ω/N) V_loc(r)``, ``(N,)``.

    Its energy with a grid density is ``sum_r ρ(r) v(r)``.
    """
    mesh = tuple(int(n) for n in mesh)
    G = gvectors(a, mesh)
    G2 = jnp.sum(G * G, axis=-1)

    def add(acc, atom):
        r, z, rl, c = atom
        phase = G @ r
        v = vloc_G(G2, z, rl, c)
        return acc + v * (jnp.cos(phase) - 1j * jnp.sin(phase)), None

    v_G, _ = jax.lax.scan(
        add, jnp.zeros(G2.shape, dtype=jnp.result_type(G2.dtype, 1j)),
        (coords, zion, rloc, cloc))
    return jnp.fft.ifftn(v_G.reshape(mesh)).real.reshape(-1)
