"""Implicit-differentiation SCF response (CPHF).

Three checks: (1) the stable occ-virt projector JVP vs finite difference; (2) the
energy gradient *through* the implicit fixed point equals the analytic Pulay-free
force (and FD), i.e. the adjoint/CPHF solve is correct; (3) the analytic
polarizability (one ``jax.jacobian`` through ``implicit_density``) equals the
finite-field polarizability.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx

from dftax.energy.xc import PBE
from dftax.system.molecule import Molecule
from dftax.basis.loader import build_basis_data
from dftax.grid import becke_grid
from dftax import KS, System, becke, scf, forces
from dftax.ks.implicit import implicit_density, _proj_from_fock
from dftax.ks.properties import polarizability

WATER = "O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043"


@pytest.mark.float64
class TestImplicit:
    def test_projector_jvp_vs_fd(self):
        rng = np.random.default_rng(0)
        n, nocc = 8, 3
        A = rng.standard_normal((n, n)); Ft = jnp.asarray(A + A.T)
        B = rng.standard_normal((n, n)); dFt = jnp.asarray(B + B.T)
        P, dP = jax.jvp(lambda F: _proj_from_fock(F, nocc), (Ft,), (dFt,))
        h = 1e-6
        fd = (_proj_from_fock(Ft + h * dFt, nocc) - _proj_from_fock(Ft - h * dFt, nocc)) / (2 * h)
        assert abs(float(jnp.trace(P)) - 2 * nocc) < 1e-10           # electron count
        assert float(jnp.max(jnp.abs(dP - fd))) < 1e-7

    def test_implicit_forward_matches_scf(self):
        mol = Molecule.from_xyz(WATER, "sto-3g")
        gc, gw = becke_grid(mol.symbols, mol.atom_coords(), 50, 110)
        ks = KS(mol, PBE(), grid=(gc, gw))
        P_impl = implicit_density(ks)
        P_ref = scf(ks, e_tol=1e-10, d_tol=1e-9).P[0]
        assert float(jnp.max(jnp.abs(P_impl - P_ref))) < 1e-7

    def test_implicit_gradient_equals_forces(self):
        mol = Molecule.from_xyz(WATER, "sto-3g")
        nocc = mol.nelectron // 2
        NR, LEB = 50, 110
        coords0 = jnp.asarray(mol.atom_coords())
        charges = jnp.asarray(mol.atom_charges(), float)
        template, aidx = build_basis_data(mol.symbols, mol.atom_coords(), mol.basis, return_atom_index=True)
        aidx = jnp.asarray(aidx)

        def energy(coords):
            basis = eqx.tree_at(lambda b: b.centers, template, coords[aidx])
            g, w = becke_grid(mol.symbols, coords, NR, LEB)
            ks = KS(
                System(basis=basis, coords=coords, charges=charges,
                       nelec=mol.nelectron),
                PBE(), grid=(g, w),
            )
            return ks.total(implicit_density(ks)[None])

        F_impl = -jax.grad(energy)(coords0)
        gc, gw = becke_grid(mol.symbols, mol.atom_coords(), NR, LEB)
        ks0 = KS(mol, PBE(), grid=(gc, gw))
        C = scf(ks0, e_tol=1e-10, d_tol=1e-9).mo_coeff[0][:, :nocc]
        F_ana = forces(mol, PBE(), (C,), grid=becke(NR, LEB))
        assert float(jnp.max(jnp.abs(F_impl - F_ana))) < 1e-6

    def test_analytic_polarizability_matches_fd(self):
        mol = Molecule.from_xyz(WATER, "sto-3g")
        kw = dict(grid=becke(60, 194), e_tol=1e-11, d_tol=1e-9)
        a_ana = np.asarray(polarizability(mol, PBE(), method="analytic", **kw))
        a_fd = np.asarray(polarizability(mol, PBE(), method="fd", field=2e-3, **kw))
        assert np.max(np.abs(a_ana - a_fd)) < 1e-4
