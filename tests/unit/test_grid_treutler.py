"""Treutler-Ahlrichs grids against PySCF's ``Grids``, where they are the default.

With the spec set to PySCF's (Treutler M4 radial mapping, Treutler cell
sizes, level-3 counts per element), dftax's grid is PySCF's: the same points
and weights to rounding, pruned or not. PySCF is the oracle only; the
molecules are built from dftax's Bohr coordinates so the two unit conversions
cannot differ.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax import KS, Molecule, becke, exact, scf
from dftax.energy.xc import PBE
from dftax.grid import becke_grid, becke_grid_size
from dftax.grid.becke import TREUTLER_XI, treutler_radial, treutler_xi
from dftax.grid.grid import _atom_shell_blocks, _element_counts

MOLECULES = {
    "water": "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
    "methane": ("C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; "
                "H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629"),
    "ammonia": ("N 0 0 0.1; H 0.94 0 -0.27; H -0.47 0.814 -0.27; "
                "H -0.47 -0.814 -0.27"),
    "h2s": "S 0 0 0.1; H 0.96 0 -0.85; H -0.96 0 -0.85",
}

# PySCF's default Grids (level 3), as a dftax spec.
PYSCF_DEFAULT = dict(n_radial="pyscf", lebedev="pyscf", radial="treutler",
                     radii_adjust="treutler")


def _pyscf_grid(symbols, coords, prune, atom_grid=None):
    from pyscf import gto
    from pyscf.dft import gen_grid

    mol = gto.M(atom=[(s, x) for s, x in zip(symbols, coords)], unit="Bohr",
                basis="sto-3g")
    g = gen_grid.Grids(mol)
    g.alignment = 0                                  # no zero-weight padding
    g.prune = gen_grid.nwchem_prune if prune == "nwchem" else None
    if atom_grid is not None:
        g.atom_grid = atom_grid
    g.build(sort_grids=False)
    return g.coords, g.weights


def _assert_same_grid(ours, ref):
    """Point sets equal as sets, and each point's weight equal, to ~1e-12."""
    from scipy.spatial import cKDTree

    (pc, pw), (rc, rw) = ours, ref
    pc, pw = np.asarray(pc), np.asarray(pw)
    assert pc.shape == rc.shape
    dist, idx = cKDTree(pc).query(rc)
    assert np.unique(idx).size == idx.size           # one-to-one
    assert dist.max() < 1e-12
    assert np.all(np.abs(pw[idx] - rw) <= 1e-12 * np.maximum(1.0, np.abs(rw)))


# ---------------------------------------------------------------------------
# Constants: xi table, radial quadrature, level-3 counts
# ---------------------------------------------------------------------------

@pytest.mark.pyscf
def test_treutler_xi_matches_pyscf():
    from pyscf.dft import radi

    ref = np.asarray(radi._treutler_ahlrichs_xi[:len(TREUTLER_XI)])
    assert np.array_equal(TREUTLER_XI, ref)


@pytest.mark.pyscf
@pytest.mark.parametrize("n", [50, 75, 80])
def test_treutler_radial_matches_pyscf(n):
    from pyscf.dft import radi

    for Z in range(1, len(TREUTLER_XI)):
        r, w = treutler_radial(n, treutler_xi(Z))
        r_ref, dr_ref = radi.treutler_ahlrichs(n, Z)
        order = np.argsort(r)
        np.testing.assert_allclose(r[order], r_ref, rtol=1e-14, atol=0)
        np.testing.assert_allclose(w[order], r_ref**2 * dr_ref, rtol=1e-13, atol=0)


@pytest.mark.pyscf
def test_pyscf_counts_are_level3():
    from pyscf.dft import gen_grid

    for Z in range(1, len(TREUTLER_XI)):
        assert _element_counts(Z, "pyscf", "pyscf") == (
            gen_grid._default_rad(Z, 3), gen_grid._default_ang(Z, 3)
        )
    assert _element_counts(8, 60, 194) == (60, 194)  # ints pass through


@pytest.mark.parametrize("kw, match", [
    (dict(radial="mura"), "unknown radial scheme"),
    (dict(n_radial="level3"), "unknown n_radial"),
    (dict(lebedev="max"), "unknown lebedev"),
])
def test_unknown_choices_raise(kw, match):
    args = dict(n_radial=50, lebedev=110, radial="becke") | kw
    with pytest.raises(ValueError, match=match):
        _atom_shell_blocks(8, args["n_radial"], args["lebedev"], None, None,
                           args["radial"])


def test_unknown_radii_adjust_raises():
    mol = Molecule.from_xyz(MOLECULES["water"], "sto-3g")
    with pytest.raises(ValueError, match="unknown radii_adjust"):
        becke_grid(mol.symbols, mol.atom_coords(), 20, 50, radii_adjust="sqrt")


# ---------------------------------------------------------------------------
# Whole molecular grids
# ---------------------------------------------------------------------------

@pytest.mark.pyscf
@pytest.mark.parametrize("prune", [None, "nwchem"])
@pytest.mark.parametrize("name", list(MOLECULES))
def test_grid_matches_pyscf_default(name, prune):
    mol = Molecule.from_xyz(MOLECULES[name], "sto-3g")
    syms, X = mol.symbols, np.asarray(mol.atom_coords())
    ours = becke_grid(syms, X, prune=prune, r_max=None, **PYSCF_DEFAULT)
    _assert_same_grid(ours, _pyscf_grid(syms, X, prune))
    # The default 45 Bohr tail cut never bites: M4 ends well inside it.
    spec = {k: PYSCF_DEFAULT[k] for k in ("n_radial", "lebedev", "radial")}
    assert becke_grid_size(syms, prune=prune, **spec) == ours[0].shape[0]


@pytest.mark.pyscf
@pytest.mark.parametrize("prune", [None, "nwchem"])
def test_grid_matches_pyscf_uniform_counts(prune):
    """An explicit ``(75, 302)`` for every element (PySCF's ``atom_grid``)."""
    mol = Molecule.from_xyz(MOLECULES["h2s"], "sto-3g")
    syms, X = mol.symbols, np.asarray(mol.atom_coords())
    ours = becke_grid(syms, X, 75, 302, prune, None, "treutler", "treutler")
    _assert_same_grid(ours, _pyscf_grid(syms, X, prune, atom_grid=(75, 302)))


def test_treutler_grid_gradient_matches_finite_difference():
    """d/dR of a grid-integrated quantity through the moving points and the
    Treutler-sized partition: the overlap of two Gaussians riding on O and H
    (the quadrature of it, so FD and AD differentiate the same function)."""
    mol = Molecule.from_xyz(MOLECULES["water"], "sto-3g")
    syms, X0 = mol.symbols, jnp.asarray(mol.atom_coords())

    def integral(X):
        gc, gw = becke_grid(syms, X, 30, 110, "nwchem", 45.0,
                            "treutler", "treutler")
        g0 = jnp.exp(-0.8 * jnp.sum((gc - X[0]) ** 2, axis=1))
        g1 = jnp.exp(-0.5 * jnp.sum((gc - X[1]) ** 2, axis=1))
        return jnp.sum(gw * g0 * g1)

    grad = np.asarray(jax.grad(integral)(X0))
    assert np.isfinite(grad).all()
    h = 1e-4
    fd = np.zeros_like(grad)
    for a in range(X0.shape[0]):
        for k in range(3):
            e = jnp.zeros_like(X0).at[a, k].set(h)
            fd[a, k] = (float(integral(X0 + e)) - float(integral(X0 - e))) / (2 * h)
    assert np.abs(grad - fd).max() < 1e-7
    assert np.abs(grad).max() > 1e-2                     # not trivially zero


# ---------------------------------------------------------------------------
# Energies (slow tier: SCF)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.pyscf
def test_scf_on_native_grid_matches_pyscf_grid():
    """The same SCF on the native grid and on PySCF's own reproduces the
    energy and the electron count to rounding, and PySCF's energy to the
    basis-precision floor (sto-3g ships 8 significant figures there)."""
    from pyscf import dft, gto

    mol = Molecule.from_xyz(MOLECULES["water"], "sto-3g")
    syms, X = mol.symbols, np.asarray(mol.atom_coords())
    pmol = gto.M(atom=[(s, x) for s, x in zip(syms, X)], unit="Bohr",
                 basis="sto-3g")
    mf = dft.RKS(pmol, xc="pbe")
    mf.verbose = 0
    e_pyscf = mf.kernel()

    grids = {
        "native": tuple(np.asarray(a) for a in becke_grid(syms, X, **PYSCF_DEFAULT)),
        "pyscf": (mf.grids.coords, mf.grids.weights),
    }
    e, n = {}, {}
    for name, (gc, gw) in grids.items():
        ks = KS(mol, PBE(), grid=(jnp.asarray(gc), jnp.asarray(gw)),
                coulomb=exact())
        res = scf(ks, e_tol=1e-11)
        assert res.converged
        rho, _ = ks.density(res.P)
        e[name] = float(res.e_tot)
        n[name] = float(jnp.sum(jnp.asarray(gw) * rho))
    assert abs(e["native"] - e["pyscf"]) < 1e-9
    assert abs(n["native"] - n["pyscf"]) < 1e-10
    assert abs(e["native"] - e_pyscf) < 1e-6

    # Through the spec (eager cutoff and all), the same energy.
    e_spec = scf(KS(mol, PBE(), grid=becke(**PYSCF_DEFAULT), coulomb=exact()),
                 e_tol=1e-11).e_tot
    assert abs(e_spec - e["native"]) < 1e-9
