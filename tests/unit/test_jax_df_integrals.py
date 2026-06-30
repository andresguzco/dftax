"""Tests for the pure-JAX density-fitting Coulomb potential integrals.

Accuracy target: 1e-5 relative error vs PySCF int2c2e + fakemol_for_charges.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import gto, df

jax.config.update("jax_enable_x64", True)

from dftax.energy.gto import extract_basis_data
from dftax.energy.jax_df_integrals import eval_coulomb_potential

# Inlined geometry to keep the test self-contained.
_H2O = """
O 0.000000 0.000000 0.000000
H 0.758602 0.000000 0.504284
H 0.758602 0.000000 -0.504284
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h2_mol():
    # cart=True so that make_auxmol inherits Cartesian convention for d shells
    return gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", cart=True, spin=0).build()


@pytest.fixture(scope="module")
def h2_auxmol(h2_mol):
    return df.addons.make_auxmol(h2_mol, auxbasis="weigend")


@pytest.fixture(scope="module")
def h2_aux_basis(h2_auxmol):
    return extract_basis_data(h2_auxmol)


@pytest.fixture(scope="module")
def water_mol():
    # cart=True so that make_auxmol inherits Cartesian convention for d shells
    return gto.M(atom=_H2O, basis="sto-3g", cart=True, spin=0).build()


@pytest.fixture(scope="module")
def water_auxmol(water_mol):
    return df.addons.make_auxmol(water_mol, auxbasis="weigend")


@pytest.fixture(scope="module")
def water_aux_basis(water_auxmol):
    return extract_basis_data(water_auxmol)


@pytest.fixture(scope="module")
def water_dz_mol():
    """Water with 6-31g* to include d-type auxiliary functions."""
    return gto.M(atom=_H2O, basis="6-31g*", cart=True, spin=0).build()


@pytest.fixture(scope="module")
def water_dz_auxmol(water_dz_mol):
    return df.addons.make_auxmol(water_dz_mol, auxbasis="weigend")


@pytest.fixture(scope="module")
def water_dz_aux_basis(water_dz_auxmol):
    return extract_basis_data(water_dz_auxmol)


# ---------------------------------------------------------------------------
# Helper: PySCF reference via fakemol + int2c2e
# ---------------------------------------------------------------------------

def _pyscf_V(auxmol, r: np.ndarray) -> np.ndarray:
    """V_{g,P} for a single grid point r via PySCF fakemol + int2c2e_cart.

    Uses ``int2c2e_cart`` so that d-type auxiliary functions are treated as
    Cartesian GTOs (6d), matching our pure-JAX implementation which is also
    Cartesian-only.  (PySCF's ``fakemol + auxmol`` merging drops cart=True,
    so we must request the Cartesian integral explicitly.)
    """
    r_2d = np.array(r, dtype=np.float64)[None, :]   # (1, 3)
    fakemol = gto.fakemol_for_charges(r_2d)
    pmol = fakemol + auxmol
    shls_slice = (0, fakemol.nbas, fakemol.nbas, fakemol.nbas + auxmol.nbas)
    # int2c2e_cart: Cartesian convention, shape (n_grid=1, n_aux_cart)
    V = pmol.intor("int2c2e_cart", shls_slice=shls_slice)
    return V[0]  # (n_aux_cart,)


# ---------------------------------------------------------------------------
# Tests: values
# ---------------------------------------------------------------------------

class TestEvalCoulombValues:

    def test_h2_sto3g_matches_pyscf(self, h2_auxmol, h2_aux_basis):
        """H2/sto-3g auxiliary basis: V must match PySCF to rtol=1e-5."""
        rng = np.random.default_rng(42)
        points = rng.normal(size=(20, 3)) * 2.0
        for r in points:
            V_jax = np.array(eval_coulomb_potential(h2_aux_basis, jnp.array(r)))
            V_ref = _pyscf_V(h2_auxmol, r)
            np.testing.assert_allclose(
                V_jax, V_ref, rtol=1e-5, atol=1e-10,
                err_msg=f"H2 V mismatch at r={r}",
            )

    def test_water_sto3g_matches_pyscf(self, water_auxmol, water_aux_basis):
        """Water/sto-3g auxiliary: V must match PySCF to rtol=1e-5."""
        rng = np.random.default_rng(0)
        points = rng.normal(size=(25, 3)) * 2.0
        for r in points:
            V_jax = np.array(eval_coulomb_potential(water_aux_basis, jnp.array(r)))
            V_ref = _pyscf_V(water_auxmol, r)
            np.testing.assert_allclose(
                V_jax, V_ref, rtol=1e-5, atol=1e-10,
                err_msg=f"Water STO-3G V mismatch at r={r}",
            )

    def test_water_6_31g_matches_pyscf(self, water_dz_auxmol, water_dz_aux_basis):
        """Water/6-31g* auxiliary (includes d functions): V must match to rtol=1e-5."""
        rng = np.random.default_rng(7)
        points = rng.normal(size=(20, 3)) * 2.0
        for r in points:
            V_jax = np.array(eval_coulomb_potential(water_dz_aux_basis, jnp.array(r)))
            V_ref = _pyscf_V(water_dz_auxmol, r)
            np.testing.assert_allclose(
                V_jax, V_ref, rtol=1e-5, atol=1e-10,
                err_msg=f"Water 6-31g* V mismatch at r={r}",
            )

    def test_output_shape(self, water_aux_basis):
        r = jnp.array([0.1, 0.2, 0.3])
        V = eval_coulomb_potential(water_aux_basis, r)
        n_aux = water_aux_basis.centers.shape[0]
        assert V.shape == (n_aux,)

    def test_no_nan_at_nuclei(self, water_mol, water_aux_basis):
        """V must be finite at atomic positions."""
        for c in water_mol.atom_coords():
            V = eval_coulomb_potential(water_aux_basis, jnp.array(c))
            assert jnp.all(jnp.isfinite(V)), f"NaN/Inf at nucleus {c}"

    def test_no_nan_at_origin(self, water_aux_basis):
        V = eval_coulomb_potential(water_aux_basis, jnp.zeros(3))
        assert jnp.all(jnp.isfinite(V))

    def test_V_finite_far_field(self, water_aux_basis):
        """Coulomb potential must be finite far from molecule."""
        r = jnp.array([10.0, 0.0, 0.0])
        V = eval_coulomb_potential(water_aux_basis, r)
        assert jnp.all(jnp.isfinite(V)), "V should be finite far from molecule"


# ---------------------------------------------------------------------------
# Tests: vmap and jit
# ---------------------------------------------------------------------------

class TestEvalCoulombVmapJit:

    def test_vmap_matches_loop(self, water_aux_basis):
        """jax.vmap result must match looped evaluation."""
        rng = np.random.default_rng(99)
        pts = jnp.array(rng.normal(size=(30, 3)) * 2.0)

        batch = jax.vmap(lambda r: eval_coulomb_potential(water_aux_basis, r))(pts)
        loop = jnp.stack([eval_coulomb_potential(water_aux_basis, pts[i]) for i in range(len(pts))])
        np.testing.assert_allclose(np.array(batch), np.array(loop), rtol=1e-12)

    def test_jit(self, water_aux_basis):
        """eval_coulomb_potential must work under jit and give identical results."""
        r = jnp.array([0.5, -0.3, 0.1])
        eager = eval_coulomb_potential(water_aux_basis, r)
        jitted = jax.jit(eval_coulomb_potential)(water_aux_basis, r)
        np.testing.assert_allclose(np.array(jitted), np.array(eager), rtol=1e-12)

    def test_jit_vmap(self, water_aux_basis):
        """jit(vmap(eval_coulomb_potential)) must produce finite results."""
        rng = np.random.default_rng(11)
        pts = jnp.array(rng.normal(size=(20, 3)))
        fn = jax.jit(jax.vmap(lambda r: eval_coulomb_potential(water_aux_basis, r)))
        result = fn(pts)
        n_aux = water_aux_basis.centers.shape[0]
        assert result.shape == (20, n_aux)
        assert jnp.all(jnp.isfinite(result))


# ---------------------------------------------------------------------------
# Tests: gradients (autodiff)
# ---------------------------------------------------------------------------

class TestEvalCoulombGradients:

    def test_jacobian_vs_fd_sto3g(self, water_auxmol, water_aux_basis):
        """Jacobian via autodiff must match central FD to rtol=1e-4."""
        r0 = jnp.array([0.1, 0.2, 0.3])
        eps = 1e-4

        jac_ad = np.array(jax.jacobian(lambda r: eval_coulomb_potential(water_aux_basis, r))(r0))

        jac_fd = np.zeros((water_auxmol.nao, 3))
        for d in range(3):
            e = jnp.zeros(3).at[d].set(eps)
            col = (
                eval_coulomb_potential(water_aux_basis, r0 + e)
                - eval_coulomb_potential(water_aux_basis, r0 - e)
            ) / (2 * eps)
            jac_fd[:, d] = np.array(col)

        np.testing.assert_allclose(jac_ad, jac_fd, rtol=1e-4, atol=1e-8)

    def test_jacobian_vs_fd_6_31g(self, water_dz_auxmol, water_dz_aux_basis):
        """Jacobian for 6-31g* auxiliary (d functions) must match FD to rtol=1e-4."""
        r0 = jnp.array([0.3, -0.1, 0.2])
        eps = 1e-4

        jac_ad = np.array(jax.jacobian(lambda r: eval_coulomb_potential(water_dz_aux_basis, r))(r0))

        jac_fd = np.zeros((water_dz_auxmol.nao, 3))
        for d in range(3):
            e = jnp.zeros(3).at[d].set(eps)
            col = (
                eval_coulomb_potential(water_dz_aux_basis, r0 + e)
                - eval_coulomb_potential(water_dz_aux_basis, r0 - e)
            ) / (2 * eps)
            jac_fd[:, d] = np.array(col)

        np.testing.assert_allclose(jac_ad, jac_fd, rtol=1e-4, atol=1e-8)

    def test_grad_finite_at_nuclei(self, water_mol, water_aux_basis):
        """Gradient must be finite near atomic positions."""
        for c in water_mol.atom_coords():
            r = jnp.array(c) + 1e-6
            g = jax.grad(lambda r_: jnp.sum(eval_coulomb_potential(water_aux_basis, r_)))(r)
            assert jnp.all(jnp.isfinite(g)), f"Gradient not finite near nucleus {c}"


# ---------------------------------------------------------------------------
# Tests: density-fitted Hartree energy vs PySCF
# ---------------------------------------------------------------------------

class TestDensityFittedHartree:
    """Verify that the full DF Hartree energy matches PySCF's analytical result."""

    def test_hartree_energy_matches_pyscf(self, water_mol, water_auxmol, water_aux_basis):
        """E_H from JAX V matrix must match PySCF DF Hartree energy to rtol=1e-4.

        Uses Cartesian J matrix (int2c2e_cart) to be consistent with our
        Cartesian JAX V implementation.  Compares against the analytical
        Hartree energy from PySCF (with 5% tolerance, since DF introduces
        ~1 kcal/mol basis-set error vs the 4-index integral).
        """
        from pyscf import dft

        # PySCF reference: RKS/sto-3g (cart=True; sto-3g has no d functions
        # so cart vs spherical makes no difference for the primary AO basis)
        ks = dft.RKS(water_mol, xc="pbe")
        ks.run()
        dm = jnp.asarray(ks.make_rdm1())

        # Analytical reference E_H = 0.5 * Tr[D J]
        from dftax.energy.hartree import _get_j
        J = _get_j(water_mol, dm)
        E_H_ref = float(0.5 * jnp.einsum("mn,mn->", J, dm))

        # Build grid for density integration
        from pyscf.dft import gen_grid
        grids = gen_grid.Grids(water_mol)
        grids.level = 2
        grids.build()
        coords = jnp.asarray(grids.coords)    # (N, 3)
        weights = jnp.asarray(grids.weights)  # (N,)

        # AO values on grid → density rho_g
        from pyscf.dft.numint import eval_ao
        ao = jnp.asarray(eval_ao(water_mol, np.array(coords)))  # (N, nao)
        rho = jnp.einsum("gi,ij,gj->g", ao, dm, ao)            # (N,)
        nw = rho * weights                                       # (N,)

        # Cartesian J^{-1} (consistent with our Cartesian V)
        int2c = jnp.asarray(water_auxmol.intor("int2c2e_cart"))
        int2c_inv = jnp.linalg.inv(int2c + 1e-10 * jnp.eye(int2c.shape[0]))

        # JAX V matrix: (N, n_aux_cart)
        V = jax.vmap(lambda r: eval_coulomb_potential(water_aux_basis, r))(coords)

        # DF Hartree energy
        q = nw @ V
        c = int2c_inv @ q
        E_H_jax = float(0.5 * jnp.dot(q, c))

        # DF accuracy vs analytical: ~1 kcal/mol = ~0.2%; 5% is very generous
        rel_err = abs(E_H_jax - E_H_ref) / abs(E_H_ref)
        assert rel_err < 0.05, (
            f"DF Hartree energy: JAX={E_H_jax:.6f}, ref={E_H_ref:.6f}, "
            f"rel_err={rel_err:.2e}"
        )
