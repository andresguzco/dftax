import jax.numpy as jnp
import numpy as np
import pytest

from dftax.energy.xc import LDA, PBE
from dftax.pbc import Cell
from dftax.pbc.lattice import reciprocal_vectors, uniform_grid, volume
from dftax.pbc.pw import monkhorst_pack, planewaves, pw_forces, pw_stress, scf_pw
from dftax.pbc.pw.basis import plan_planewaves
from dftax.pbc.pw.energy import (
    apply_h, build_model, initial_density, potential, projector_matrices,
    total_energy, wavevectors,
)
from dftax.pbc.pw.projectors import channels, projectors
from dftax.pbc.pw.solve import lobpcg

ALAT = 5.431
A_SI = 0.5 * ALAT * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
SI2 = f"Si 0 0 0; Si {ALAT / 4 + 0.05} {ALAT / 4 - 0.03} {ALAT / 4 + 0.04}"


def test_projectors_match_real_space_gaussians():
    from dftax.pbc.bloch import ao_on_grid, plan_values
    from dftax.pbc.gth import projector_basis

    A = np.array([[7.0, 0.0, 0.0], [1.0, 6.5, 0.0], [0.5, -1.0, 7.5]])
    cell = Cell.from_xyz("Si 0.5 0.4 0.3; O 3.4 1.1 0.9", A, "gth-szv",
                         "gth-pade", unit="bohr")
    a = jnp.asarray(cell.a)
    vol = volume(a)
    plan = plan_planewaves(cell.a, planewaves(6.0))
    n = plan.npw[0]
    q = (plan.miller[0, :n] + plan.kfrac[0]) @ reciprocal_vectors(a)
    chans, D = channels(cell.potentials)
    P = projectors(q, jnp.asarray(cell.atom_coords()), chans, vol)
    V = P.conj().T @ D @ P

    pbasis, H = projector_basis(cell.potentials, cell.atom_coords())
    mesh = (80, 80, 84)
    pts, _ = uniform_grid(a, mesh)
    vals, _ = ao_on_grid(pbasis, a, pts, plan_values(pbasis, cell.a, pts))
    N = int(np.prod(mesh))
    F = jnp.fft.fftn(vals.reshape(*mesh, -1), axes=(0, 1, 2)).reshape(N, -1)
    m = np.mod(np.asarray(plan.miller[0, :n]).astype(int), mesh)
    idx = (m[:, 0] * mesh[1] + m[:, 1]) * mesh[2] + m[:, 2]
    Pnum = (F[idx] * jnp.sqrt(vol) / N).conj().T
    Vnum = Pnum.conj().T @ jnp.asarray(H) @ Pnum
    np.testing.assert_allclose(V, Vnum, atol=1e-8)


@pytest.mark.parametrize("xc", [LDA(), PBE()], ids=["lda", "pbe"])
def test_band_energies_are_energy_derivatives(xc):
    cell = Cell.from_xyz(SI2, A_SI, "gth-szv", "gth-pade")
    model = build_model(cell, xc, planewaves(6.0), *monkhorst_pack((2, 1, 1)))
    coords, a = jnp.asarray(cell.atom_coords()), jnp.asarray(cell.a)
    nk, npw, nb = model.miller.shape[0], model.miller.shape[1], model.nbands
    rng = np.random.default_rng(1)
    X = rng.normal(size=(nk, npw, nb)) + 1j * rng.normal(size=(nk, npw, nb))
    X *= np.asarray(model.mask)[:, :, None]
    C = jnp.asarray(np.stack([np.linalg.qr(x)[0] for x in X]))
    occ = jnp.asarray(rng.uniform(0.5, 2.0, size=(nk, nb)))

    from dftax.pbc.pw.energy import density
    rho = density(model, C, occ, a)
    vr = potential(model, rho, coords, a)
    q = wavevectors(model, a)
    kin = 0.5 * jnp.sum(q * q, axis=-1) * model.mask
    P = projector_matrices(model, coords, a)
    for k, nband in [(0, 0), (1, 3)]:
        HC = apply_h(C[k], kin[k], vr, model.fft_idx[k], model.mask[k],
                     model.mesh, P[k], model.D)
        eps = float(jnp.vdot(C[k][:, nband], HC[:, nband]).real)
        h = 1e-4
        ep = total_energy(model, C, occ.at[k, nband].add(h), coords, a)
        em = total_energy(model, C, occ.at[k, nband].add(-h), coords, a)
        np.testing.assert_allclose(float(ep - em) / (2 * h),
                                   float(model.weights[k]) * eps, atol=1e-7)


def test_eigensolver_matches_dense_diagonalization():
    cell = Cell.from_xyz(SI2, A_SI, "gth-szv", "gth-pade")
    model = build_model(cell, LDA(), planewaves(5.0), *monkhorst_pack((2, 1, 1)))
    coords, a = jnp.asarray(cell.atom_coords()), jnp.asarray(cell.a)
    vr = potential(model, initial_density(model, coords, a), coords, a)
    q = wavevectors(model, a)
    kin = 0.5 * jnp.sum(q * q, axis=-1) * model.mask
    P = projector_matrices(model, coords, a)
    k, n = 1, model.npw[1]
    npw, nb = model.miller.shape[1], model.nbands
    Hd = apply_h(jnp.eye(npw, dtype=jnp.complex128), kin[k], vr,
                 model.fft_idx[k], model.mask[k], model.mesh, P[k], model.D)
    Hd = np.asarray(Hd)[:n, :n]
    np.testing.assert_allclose(Hd, Hd.conj().T, atol=1e-12)
    ref = np.linalg.eigvalsh(Hd)[:nb]

    rng = np.random.default_rng(0)
    X = (rng.normal(size=(npw, nb)) + 1j * rng.normal(size=(npw, nb)))
    X = np.linalg.qr(X * np.asarray(model.mask[k])[:, None])[0]
    _, e, _ = lobpcg(jnp.asarray(X), kin[k], vr, model.fft_idx[k],
                     model.mask[k], model.mesh, P[k], model.D, 60)
    np.testing.assert_allclose(np.asarray(e)[:model.nocc], ref[:model.nocc],
                               atol=1e-9)


@pytest.fixture(scope="module")
def si2_scf():
    cell = Cell.from_xyz(SI2, A_SI, "gth-szv", "gth-pade")
    res = scf_pw(cell, LDA(), planewaves(8.0), kpts=monkhorst_pack((2, 2, 2)),
                 e_tol=1e-11, d_tol=1e-8)
    assert res.converged
    return cell, res


def _energy_at(coords, a, model):
    cell = Cell(["Si", "Si"], coords, a, "gth-szv", "gth-pade")
    res = scf_pw(cell, LDA(), planewaves(8.0), model=model, e_tol=1e-11,
                 d_tol=1e-8)
    assert res.converged
    return res.e_tot


def test_forces_match_finite_differences(si2_scf):
    cell, res = si2_scf
    F = np.asarray(pw_forces(cell, res))
    h = 1e-3
    for atom, axis in [(1, 0), (0, 2)]:
        d = np.zeros((2, 3))
        d[atom, axis] = h
        ep = _energy_at(cell.atom_coords() + d, cell.a, res.model)
        em = _energy_at(cell.atom_coords() - d, cell.a, res.model)
        np.testing.assert_allclose(F[atom, axis], -(ep - em) / (2 * h), atol=1e-5)


def test_stress_matches_finite_differences(si2_scf):
    cell, res = si2_scf
    sigma = np.asarray(pw_stress(cell, res))
    vol = abs(np.linalg.det(cell.a))
    h = 1e-3
    for i, j in [(0, 0), (0, 1)]:
        eps = np.zeros((3, 3))
        eps[i, j] = eps[j, i] = h
        e = [_energy_at(cell.atom_coords() @ (np.eye(3) + s * eps).T,
                        cell.a @ (np.eye(3) + s * eps).T, res.model)
             for s in (1, -1)]
        np.testing.assert_allclose(sigma[i, j] * (1 if i == j else 2),
                                   (e[0] - e[1]) / (2 * h) / vol, atol=1e-7)


def test_smeared_stress_matches_finite_differences():
    import sys
    from pathlib import Path

    from dftax import fermi
    from dftax.pbc import kmesh, stress

    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "pbc"))
    from systems import ANGSTROM_TO_BOHR, systems

    sp = systems()["al4"]
    coords, a = sp.cart * ANGSTROM_TO_BOHR, sp.a * ANGSTROM_TO_BOHR

    def solve(R, L, model=None):
        cell = Cell(list(sp.symbols), R, L, "gth-szv", "gth-pade")
        res = scf_pw(cell, LDA(), planewaves(6.0), kpts=kmesh(2, 2, 2),
                     smearing=fermi(sigma=0.01), model=model, e_tol=1e-11,
                     d_tol=1e-8)
        assert res.converged
        return cell, res

    cell, res = solve(coords, a)
    sigma = np.asarray(stress(cell, LDA(), res))
    h = 1e-3
    eps = np.zeros((3, 3))
    eps[0, 0] = h
    e = [solve(coords @ (np.eye(3) + s * eps).T, a @ (np.eye(3) + s * eps).T,
               res.model)[1].e_tot for s in (1, -1)]
    np.testing.assert_allclose(sigma[0, 0], (e[0] - e[1]) / (2 * h)
                               / abs(np.linalg.det(a)), atol=1e-7)
