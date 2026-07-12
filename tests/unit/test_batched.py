"""Batched (vmap-over-geometries) interface: batched == serial, energies + forces.

Small grids: the batched-vs-serial check is grid-independent (both paths use the
same grid), and convergence to a few µHa suffices for the equality assertions.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from dftax.energy.xc import PBE
from dftax.system.molecule import Molecule
from dftax import KS, becke, scf, forces, scf_batched, BatchedResult

NR, LEB = 35, 50
GRID = becke(NR, LEB)


@pytest.mark.float64
class TestBatched:
    def test_rks_energy_matches_serial(self):
        mol = Molecule.from_xyz("O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50", "sto-3g")
        c0 = jnp.asarray(mol.atom_coords())
        batch = jnp.stack([c0, c0.at[1, 2].add(0.05), c0.at[2, 0].add(-0.04)])
        rb = scf_batched(mol, batch, PBE(), grid=GRID)
        assert isinstance(rb, BatchedResult)
        # orbital-sized fields are opt-in (O(B*nspin*nao^2) memory)
        assert rb.P is None and rb.mo_coeff is None and rb.mo_energy is None
        with pytest.raises(TypeError):
            forces(mol, PBE(), rb, grid=GRID)           # batched result rejected
        assert bool(jnp.all(rb.converged))
        for b in range(batch.shape[0]):
            m = Molecule(mol.symbols, np.asarray(batch[b]), mol.basis)
            es = scf(KS(m, PBE(), grid=GRID)).e_tot
            assert abs(float(rb.e_tot[b]) - es) < 1e-9

    def test_uks_energy_matches_serial(self):
        ch3 = Molecule.from_xyz(
            "C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0", "sto-3g", spin=1
        )
        cc = jnp.asarray(ch3.atom_coords())
        batch = jnp.stack([cc, cc.at[1, 1].add(0.05)])
        ub = scf_batched(ch3, batch, PBE(), spin=ch3.spin, grid=GRID)
        assert bool(jnp.all(ub.converged))
        for b in range(batch.shape[0]):
            m = Molecule(ch3.symbols, np.asarray(batch[b]), ch3.basis, spin=1)
            es = scf(KS(m, PBE(), grid=GRID, spin=m.spin)).e_tot
            assert abs(float(ub.e_tot[b]) - es) < 1e-9

    def test_rks_forces_match_serial(self):
        mol = Molecule.from_xyz("O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50", "sto-3g")
        nocc = mol.nelectron // 2
        c0 = jnp.asarray(mol.atom_coords())
        batch = jnp.stack([c0, c0.at[1, 2].add(0.05)])
        rb = scf_batched(mol, batch, PBE(), forces=True, grid=GRID)
        assert rb.forces.shape == batch.shape
        for b in range(batch.shape[0]):
            m = Molecule(mol.symbols, np.asarray(batch[b]), mol.basis)
            r = scf(KS(m, PBE(), grid=GRID))
            Fser = forces(m, PBE(), (r.mo_coeff[0][:, :nocc],), grid=GRID)
            assert float(jnp.max(jnp.abs(rb.forces[b] - Fser))) < 1e-9

    def test_uks_forces_match_serial(self):
        ch3 = Molecule.from_xyz(
            "C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0", "sto-3g", spin=1
        )
        na = (ch3.nelectron + 1) // 2
        nb = ch3.nelectron - na
        cc = jnp.asarray(ch3.atom_coords())
        ub = scf_batched(ch3, jnp.stack([cc]), PBE(), spin=ch3.spin, forces=True, grid=GRID)
        m = Molecule(ch3.symbols, np.asarray(cc), ch3.basis, spin=1)
        r = scf(KS(m, PBE(), grid=GRID, spin=m.spin))
        Fser = forces(
            m, PBE(), (r.mo_coeff[0][:, :na], r.mo_coeff[1][:, :nb]), grid=GRID
        )
        assert float(jnp.max(jnp.abs(ub.forces[0] - Fser))) < 1e-9

    def test_scf_batched_spin_dispatch(self):
        # Closed shell + inferred spin -> a single restricted channel.
        mol = Molecule.from_xyz("O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50", "sto-3g")
        c0 = jnp.asarray(mol.atom_coords())
        rb = scf_batched(mol, jnp.stack([c0]), PBE(), grid=GRID, return_orbitals=True)
        assert len(rb.nocc) == 1
        assert rb.P.shape[:2] == (1, 1)                 # (batch, nspin, ...)
        # Open shell: inferred spin == explicit spin=, identical computation.
        rad = Molecule.from_xyz("Li 0 0 0", "sto-3g", spin=1)
        cl = jnp.asarray(rad.atom_coords())
        r_auto = scf_batched(rad, jnp.stack([cl]), PBE(), grid=GRID)
        r_uks = scf_batched(rad, jnp.stack([cl]), PBE(), spin=rad.spin, grid=GRID)
        assert len(r_auto.nocc) == 2
        assert abs(float(r_auto.e_tot[0]) - float(r_uks.e_tot[0])) < 1e-12
