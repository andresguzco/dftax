"""The Gaussian periodic engine at the Gamma point against PySCF's ``pbc``
module, on a triclinic two-element cell with d functions (``gth-dzvp``)."""

import jax.numpy as jnp
import numpy as np
import pytest

from dftax.basis.loader import build_basis_data
from dftax.pbc import Cell, uniform_grid
from dftax.pbc.bloch import ao_on_grid, overlap_kinetic, pair_images, value_images

A = np.array([[9.0, 0.0, 0.0], [2.0, 8.5, 0.0], [1.0, -1.5, 10.0]])   # Bohr
ATOMS = "Si 0.5 0.4 0.3; O 3.4 1.1 0.9; O 1.0 4.2 5.6"                # Bohr
MESH = (9, 10, 12)


@pytest.fixture(scope="module")
def cell():
    return Cell.from_xyz(ATOMS, A, "gth-dzvp", "gth-pade", unit="bohr")


@pytest.fixture(scope="module")
def basis(cell):
    return build_basis_data(cell.symbols, cell.atom_coords(), cell.basis,
                            spherical=True)


@pytest.fixture(scope="module")
def pcell():
    from pyscf.pbc import gto

    c = gto.Cell()
    c.atom, c.a, c.unit = ATOMS, A, "B"
    c.basis, c.pseudo = "gth-dzvp", "gth-pade"
    c.mesh = list(MESH)
    c.precision = 1e-12          # PySCF's lattice sums converge to this
    c.verbose = 0
    c.build()
    return c


def test_overlap_and_kinetic_match_pyscf(cell, basis, pcell):
    a = jnp.asarray(cell.a)
    S, T = overlap_kinetic(basis, a, pair_images(basis, cell.a))
    np.testing.assert_allclose(S, pcell.pbc_intor("int1e_ovlp"), atol=1e-9)
    np.testing.assert_allclose(T, pcell.pbc_intor("int1e_kin"), atol=1e-9)


def test_ao_values_match_pyscf(cell, basis, pcell):
    from pyscf.pbc.dft import numint

    a = jnp.asarray(cell.a)
    coords, _ = uniform_grid(a, MESH)
    ao, dao = ao_on_grid(basis, a, value_images(basis, cell.a), coords)
    ref = numint.eval_ao(pcell, np.asarray(coords), deriv=1)     # (4, ng, nao)
    np.testing.assert_allclose(ao, ref[0], atol=1e-9)
    np.testing.assert_allclose(np.moveaxis(np.asarray(dao), -1, 0), ref[1:],
                               atol=1e-9)


def _pyscf_local_and_nonlocal(pcell, coords):
    """PySCF's pseudopotential matrices in the engine's formulation: the
    local part by quadrature on the uniform grid, the nonlocal part by
    analytic projector integrals (PySCF's FFT class takes the nonlocal part
    in reciprocal space on the mesh instead, which a coarse mesh truncates)."""
    from pyscf.pbc import tools
    from pyscf.pbc.dft import numint
    from pyscf.pbc.gto import pseudo as ppseudo
    from pyscf.pbc.gto.pseudo import pp_int

    mesh = list(pcell.mesh)
    v_G = -np.einsum("ij,ij->j", pcell.get_SI(mesh=mesh),
                     ppseudo.get_vlocG(pcell, pcell.get_Gv(mesh)))
    v_R = tools.ifft(v_G, mesh).real
    ao = numint.eval_ao(pcell, np.asarray(coords))
    vnl = np.asarray(pp_int.get_pp_nl(pcell))
    assert np.abs(vnl.imag).max() < 1e-12
    return np.einsum("gi,g,gj->ij", ao, v_R, ao), vnl.real


def test_pseudopotential_matrices_match_pyscf(cell, basis, pcell):
    from dftax.pbc.bloch import cross_overlap
    from dftax.pbc.electrostatics import local_potential
    from dftax.pbc.gth import local_parameters, projector_basis

    a = jnp.asarray(cell.a)
    coords, _ = uniform_grid(a, MESH)
    vloc_ref, vnl_ref = _pyscf_local_and_nonlocal(pcell, coords)

    pbasis, H = projector_basis(cell.potentials, cell.atom_coords())
    B = cross_overlap(basis, pbasis, a, pair_images(basis, cell.a, other=pbasis))
    np.testing.assert_allclose(B @ H @ B.T, vnl_ref, atol=1e-9)

    ao, _ = ao_on_grid(basis, a, value_images(basis, cell.a), coords)
    zion, rloc, cloc = map(jnp.asarray, local_parameters(cell.potentials))
    v = local_potential(jnp.asarray(cell.atom_coords()), a, MESH, zion, rloc, cloc)
    np.testing.assert_allclose((ao * v[:, None]).T @ ao, vloc_ref, atol=1e-9)


def test_si2_energy_matches_pyscf():
    """Diamond silicon, primitive cell: every energy term at PySCF's
    converged density, then the SCF energy itself."""
    from pyscf.pbc import dft as pdft, gto

    from dftax import KS, scf
    from dftax.energy.xc import LDA
    from dftax.pbc import uniform

    alat = 5.431
    a = 0.5 * alat * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    atoms = f"Si 0 0 0; Si {alat / 4} {alat / 4} {alat / 4}"
    mesh = (15, 15, 15)

    pc = gto.Cell()
    pc.atom, pc.a, pc.unit = atoms, a, "A"
    pc.basis, pc.pseudo = "gth-dzvp", "gth-pade"
    pc.mesh, pc.precision, pc.verbose = list(mesh), 1e-12, 0
    pc.build()

    cell = Cell.from_xyz(atoms, a, "gth-dzvp", "gth-pade")
    ks = KS(cell, LDA(), grid=uniform(mesh=mesh))

    coords, _ = uniform_grid(jnp.asarray(cell.a), mesh)
    vloc, vnl = _pyscf_local_and_nonlocal(pc, coords)
    h_ref = pc.pbc_intor("int1e_kin") + vloc + vnl
    np.testing.assert_allclose(ks.hcore, h_ref, atol=1e-9)

    mf = pdft.RKS(pc)
    mf.xc = "lda,vwn"
    mf.get_hcore = lambda *args, **kwargs: h_ref
    mf.conv_tol = 1e-11
    e_ref = mf.kernel()
    assert mf.converged
    dm = mf.make_rdm1()
    P = jnp.asarray(dm.real)[None]

    s = mf.scf_summary
    np.testing.assert_allclose(jnp.sum(P[0] * ks.hcore), s["e1"], atol=1e-9)
    np.testing.assert_allclose(ks.coulomb.energy(P, ks.S, ks.nocc), s["coul"],
                               atol=1e-9)
    np.testing.assert_allclose(ks.xc_term.energy(P), s["exc"], atol=1e-9)
    np.testing.assert_allclose(ks.e_nn, s["nuc"], atol=1e-9)
    np.testing.assert_allclose(ks.total(P), e_ref, atol=1e-8)

    res = scf(ks, e_tol=1e-10, d_tol=1e-7)
    assert res.converged
    np.testing.assert_allclose(res.e_tot, e_ref, atol=1e-8)


def test_streamed_grid_matches_held_values(cell):
    """The chunked pass over the grid gives the energy and the Fock matrix
    of the held-values build (PBE, so the density gradient is exercised)."""
    from dftax import KS
    from dftax.energy.xc import PBE
    from dftax.pbc import uniform

    held = KS(cell, PBE(), grid=uniform(mesh=MESH, chunk=None))
    streamed = KS(cell, PBE(), grid=uniform(mesh=MESH, chunk=256))
    C = np.random.default_rng(0).normal(size=(held.S.shape[0], held.nocc[0]))
    P = C @ C.T
    P = jnp.asarray(P * held.nelec / np.sum(P * np.asarray(held.S)))[None]
    e0, F0 = held.energy_and_fock(P)
    e1, F1 = streamed.energy_and_fock(P)
    np.testing.assert_allclose(e1, e0, atol=1e-10)
    np.testing.assert_allclose(F1, F0, atol=1e-9)


def test_spin_polarized_energy_matches_pyscf():
    """Two spin channels through the periodic terms: the energy of a triplet
    density of the silicon cell, at PySCF's own density matrices (a fixed
    density, so neither code's SCF path enters)."""
    from pyscf.pbc import dft as pdft, gto

    from dftax import KS
    from dftax.energy.xc import LDA
    from dftax.pbc import uniform

    alat = 5.431
    a = 0.5 * alat * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    atoms = f"Si 0 0 0; Si {alat / 4} {alat / 4} {alat / 4}"
    mesh = (15, 15, 15)

    pc = gto.Cell()
    pc.atom, pc.a, pc.unit, pc.spin = atoms, a, "A", 2
    pc.basis, pc.pseudo = "gth-szv", "gth-pade"
    pc.mesh, pc.precision, pc.verbose = list(mesh), 1e-12, 0
    pc.build()

    cell = Cell.from_xyz(atoms, a, "gth-szv", "gth-pade", spin=2)
    ks = KS(cell, LDA(), grid=uniform(mesh=mesh))
    assert ks.nocc == (5, 3)

    coords, _ = uniform_grid(jnp.asarray(cell.a), mesh)
    vloc, vnl = _pyscf_local_and_nonlocal(pc, coords)
    h_ref = pc.pbc_intor("int1e_kin") + vloc + vnl
    mf = pdft.UKS(pc)
    mf.xc = "lda,vwn"
    mf.get_hcore = lambda *args, **kwargs: h_ref
    mf.max_cycle = 8
    mf.kernel()
    dm = np.asarray(mf.make_rdm1()).real                   # (2, nao, nao)
    np.testing.assert_allclose(ks.total(jnp.asarray(dm)), mf.energy_tot(dm=dm),
                               atol=1e-8)
