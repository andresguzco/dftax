"""Batched (vmap-over-geometries) interface: batched == serial, energies + forces.

Small grids: the batched-vs-serial check is grid-independent (both paths use the
same grid), and convergence to a few µHa suffices for the equality assertions.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from dftax.energy.xc import PBE
from dftax.system.molecule import Molecule
from dftax import (
    run_rks, run_uks, rks_forces, uks_forces,
    run_rks_batched, run_uks_batched, run_ks_batched, BatchedResult,
)

NR, LEB = 35, 50
KW = dict(n_radial=NR, lebedev=LEB)


@pytest.mark.float64
class TestBatched:
    def test_rks_energy_matches_serial(self):
        mol = Molecule.from_xyz("O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50", "sto-3g")
        c0 = jnp.asarray(mol.atom_coords())
        batch = jnp.stack([c0, c0.at[1, 2].add(0.05), c0.at[2, 0].add(-0.04)])
        rb = run_rks_batched(mol, batch, PBE(), **KW)
        assert isinstance(rb, BatchedResult)
        assert bool(jnp.all(rb.converged))
        for b in range(batch.shape[0]):
            m = Molecule(mol.symbols, np.asarray(batch[b]), mol.basis)
            es = run_rks(m, PBE(), **KW).e_tot
            assert abs(float(rb.e_tot[b]) - es) < 1e-9

    def test_uks_energy_matches_serial(self):
        ch3 = Molecule.from_xyz(
            "C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0", "sto-3g", spin=1
        )
        cc = jnp.asarray(ch3.atom_coords())
        batch = jnp.stack([cc, cc.at[1, 1].add(0.05)])
        ub = run_uks_batched(ch3, batch, PBE(), **KW)
        assert bool(jnp.all(ub.converged))
        for b in range(batch.shape[0]):
            m = Molecule(ch3.symbols, np.asarray(batch[b]), ch3.basis, spin=1)
            es = run_uks(m, PBE(), **KW).e_tot
            assert abs(float(ub.e_tot[b]) - es) < 1e-9

    def test_rks_forces_match_serial(self):
        mol = Molecule.from_xyz("O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50", "sto-3g")
        nocc = mol.nelectron // 2
        c0 = jnp.asarray(mol.atom_coords())
        batch = jnp.stack([c0, c0.at[1, 2].add(0.05)])
        rb = run_rks_batched(mol, batch, PBE(), forces=True, **KW)
        assert rb.forces.shape == batch.shape
        for b in range(batch.shape[0]):
            m = Molecule(mol.symbols, np.asarray(batch[b]), mol.basis)
            r = run_rks(m, PBE(), **KW)
            Fser = rks_forces(m, PBE(), r.mo_coeff[:, :nocc], **KW)
            assert float(jnp.max(jnp.abs(rb.forces[b] - Fser))) < 1e-9

    def test_uks_forces_match_serial(self):
        ch3 = Molecule.from_xyz(
            "C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0", "sto-3g", spin=1
        )
        na = (ch3.nelectron + 1) // 2
        nb = ch3.nelectron - na
        cc = jnp.asarray(ch3.atom_coords())
        ub = run_uks_batched(ch3, jnp.stack([cc]), PBE(), forces=True, **KW)
        m = Molecule(ch3.symbols, np.asarray(cc), ch3.basis, spin=1)
        r = run_uks(m, PBE(), **KW)
        Fser = uks_forces(m, PBE(), r.mo_coeff[0][:, :na], r.mo_coeff[1][:, :nb], **KW)
        assert float(jnp.max(jnp.abs(ub.forces[0] - Fser))) < 1e-9

    def test_run_ks_batched_dispatch(self):
        mol = Molecule.from_xyz("O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50", "sto-3g")
        c0 = jnp.asarray(mol.atom_coords())
        rb = run_ks_batched(mol, jnp.stack([c0]), PBE(), **KW)
        rr = run_rks_batched(mol, jnp.stack([c0]), PBE(), **KW)
        assert abs(float(rb.e_tot[0]) - float(rr.e_tot[0])) < 1e-12
