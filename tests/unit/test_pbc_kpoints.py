"""The Gaussian engine at k-points against PySCF's ``pbc`` (KRKS): Bloch
matrices and AO values at a general k-point of a triclinic cell, then the
energy terms and the SCF energy of silicon on a k-mesh."""

import jax.numpy as jnp
import numpy as np
import pytest

from dftax.basis.loader import build_basis_data
from dftax.pbc import Cell, uniform_grid
from dftax.pbc.bloch import (
    ao_on_grid_k, cross_overlap_k, overlap_kinetic_k, plan_pairs, plan_values,
)

A = np.array([[9.0, 0.0, 0.0], [2.0, 8.5, 0.0], [1.0, -1.5, 10.0]])   # Bohr
ATOMS = "Si 0.5 0.4 0.3; O 3.4 1.1 0.9; O 1.0 4.2 5.6"                # Bohr
MESH = (9, 10, 12)
KFRAC = np.array([[0.25, -0.5, 0.125], [0.0, 0.0, 0.0]])


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
    c.mesh, c.precision, c.verbose = list(MESH), 1e-12, 0
    c.build()
    return c


def test_bloch_matrices_match_pyscf(cell, basis, pcell):
    from pyscf.pbc.gto.pseudo import pp_int

    from dftax.pbc.gth import projector_basis

    a = jnp.asarray(cell.a)
    kabs = pcell.get_abs_kpts(KFRAC)
    S, T = overlap_kinetic_k(basis, a, plan_pairs(basis, cell.a), KFRAC)
    S_ref = np.asarray(pcell.pbc_intor("int1e_ovlp", kpts=kabs))
    T_ref = np.asarray(pcell.pbc_intor("int1e_kin", kpts=kabs))
    np.testing.assert_allclose(S, S_ref, atol=1e-9)
    np.testing.assert_allclose(T, T_ref, atol=1e-9)

    pbasis, H = projector_basis(cell.potentials, cell.atom_coords())
    B = cross_overlap_k(basis, pbasis, a, plan_pairs(basis, cell.a, other=pbasis),
                        KFRAC)
    vnl = B @ jnp.asarray(H) @ jnp.conj(jnp.swapaxes(B, -1, -2))
    np.testing.assert_allclose(vnl, np.asarray(pp_int.get_pp_nl(pcell, kabs)),
                               atol=1e-9)


def test_bloch_ao_values_match_pyscf(cell, basis, pcell):
    from pyscf.pbc.dft import numint

    a = jnp.asarray(cell.a)
    coords, _ = uniform_grid(a, MESH)
    v, g = ao_on_grid_k(basis, a, coords, plan_values(basis, cell.a, coords),
                        KFRAC)
    for i, k in enumerate(pcell.get_abs_kpts(KFRAC)):
        ref = numint.eval_ao(pcell, np.asarray(coords), kpt=k, deriv=1)
        np.testing.assert_allclose(v[i], ref[0], atol=1e-9)
        np.testing.assert_allclose(np.moveaxis(np.asarray(g[i]), -1, 0),
                                   ref[1:], atol=1e-9)


def _pyscf_hcore_k(pc, coords, kpts):
    """PySCF's core Hamiltonian per k in the engine's formulation (the local
    part by quadrature on the uniform grid, the nonlocal part analytic), and
    its kinetic + nonlocal part (the k-point engine keeps the local part in
    its grid energy)."""
    from pyscf.pbc import tools
    from pyscf.pbc.dft import numint
    from pyscf.pbc.gto import pseudo as ppseudo
    from pyscf.pbc.gto.pseudo import pp_int

    mesh = list(pc.mesh)
    v_G = -np.einsum("ij,ij->j", pc.get_SI(mesh=mesh),
                     ppseudo.get_vlocG(pc, pc.get_Gv(mesh)))
    v_R = tools.ifft(v_G, mesh).real
    vnl = np.asarray(pp_int.get_pp_nl(pc, kpts))
    kin = np.asarray(pc.pbc_intor("int1e_kin", kpts=kpts))
    out = []
    for i, k in enumerate(kpts):
        ao = numint.eval_ao(pc, np.asarray(coords), kpt=k)
        out.append(kin[i] + np.einsum("gi,g,gj->ij", ao.conj(), v_R, ao) + vnl[i])
    return np.asarray(out), kin + vnl


def test_si2_kmesh_energy_matches_pyscf():
    """Silicon on a 2x2x2 k-mesh: the core Hamiltonian of every k, every
    energy term at PySCF's converged density, and the SCF energy."""
    from pyscf.pbc import dft as pdft, gto

    from dftax.energy.xc import LDA
    from dftax.pbc import uniform
    from dftax.pbc.kpoints import kmesh, kpoint_ks, scf_k

    alat = 5.431
    a = 0.5 * alat * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    atoms = f"Si 0 0 0; Si {alat / 4} {alat / 4} {alat / 4}"
    mesh = (15, 15, 15)
    kfrac, kw = kmesh(2, 2, 2)

    pc = gto.Cell()
    pc.atom, pc.a, pc.unit = atoms, a, "A"
    pc.basis, pc.pseudo = "gth-szv", "gth-pade"
    pc.mesh, pc.precision, pc.verbose = list(mesh), 1e-12, 0
    pc.build()
    kabs = pc.get_abs_kpts(kfrac)

    cell = Cell.from_xyz(atoms, a, "gth-szv", "gth-pade")
    ks = kpoint_ks(cell, LDA(), grid=uniform(mesh=mesh), kpts=(kfrac, kw))
    coords, _ = uniform_grid(jnp.asarray(cell.a), mesh)
    h_ref, h_kin_nl = _pyscf_hcore_k(pc, coords, kabs)
    np.testing.assert_allclose(ks.hcore, h_kin_nl, atol=1e-9)

    mf = pdft.KRKS(pc, kabs)
    mf.xc = "lda,vwn"
    mf.get_hcore = lambda *args, **kwargs: h_ref
    mf.conv_tol = 1e-11
    e_ref = mf.kernel()
    assert mf.converged
    P = jnp.asarray(np.asarray(mf.make_rdm1()))
    np.testing.assert_allclose(ks.total(P), e_ref, atol=1e-8)

    res = scf_k(ks, e_tol=1e-10, d_tol=1e-7)
    assert res.converged
    np.testing.assert_allclose(res.e_tot, e_ref, atol=1e-8)


def test_kpoint_forces_and_stress_match_finite_differences():
    """Rattled silicon on a 2x2x2 k-mesh with Fermi smearing (fractional
    occupations, the free energy): forces and stress against finite
    differences of converged free energies, through the common verbs."""
    from dftax import fermi, forces, scf
    from dftax.energy.xc import LDA
    from dftax.pbc import kmesh, kpoint_ks, stress, uniform

    alat = 5.431 * 1.8897261246
    a = 0.5 * alat * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    coords = np.array([[0.0, 0.0, 0.0],
                       [0.25 * alat + 0.10, 0.25 * alat - 0.05, 0.25 * alat + 0.07]])
    grid, kpts = uniform(mesh=(15, 15, 15)), kmesh(2, 2, 2)

    def solve(R, L):
        cell = Cell(["Si", "Si"], R, L, "gth-szv", "gth-pade")
        res = scf(kpoint_ks(cell, LDA(), grid=grid, kpts=kpts),
                  smearing=fermi(sigma=0.02), e_tol=1e-12, d_tol=1e-9)
        assert res.converged
        return cell, res

    cell, res = solve(coords, a)
    F = np.asarray(forces(cell, LDA(), res, grid=grid))
    sigma_t = np.asarray(stress(cell, LDA(), res, grid=grid))
    h = 1e-3
    d = np.zeros((2, 3))
    d[1, 0] = h
    fd = -(solve(coords + d, a)[1].e_tot - solve(coords - d, a)[1].e_tot) / (2 * h)
    np.testing.assert_allclose(F[1, 0], fd, atol=2e-6)
    eps = np.zeros((3, 3))
    eps[0, 1] = eps[1, 0] = h
    e = [solve(coords @ (np.eye(3) + s * eps).T, a @ (np.eye(3) + s * eps).T)[1].e_tot
         for s in (1, -1)]
    np.testing.assert_allclose(2 * sigma_t[0, 1],
                               (e[0] - e[1]) / (2 * h) / abs(np.linalg.det(a)),
                               atol=1e-7)
