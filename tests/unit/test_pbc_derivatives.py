"""Forces and stress of the Gaussian periodic engine against finite
differences of the converged energy, on a rattled primitive silicon cell.

The comparison with PySCF's own derivatives (its stress is partly a finite
difference itself) is the baseline gate, ``scripts/pbc/run_dftax.py``."""

import numpy as np
import pytest

from dftax import KS, forces, scf
from dftax.energy.xc import LDA
from dftax.pbc import Cell, stress, uniform

ALAT = 5.431 * 1.8897261246                       # Bohr
A = 0.5 * ALAT * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
COORDS = np.array([[0.0, 0.0, 0.0],
                   [0.25 * ALAT + 0.10, 0.25 * ALAT - 0.05, 0.25 * ALAT + 0.07]])
GRID = uniform(mesh=(15, 15, 15))
H = 1e-3


def _solve(coords, a):
    cell = Cell(["Si", "Si"], coords, a, "gth-szv", "gth-pade")
    res = scf(KS(cell, LDA(), grid=GRID), e_tol=1e-11, d_tol=1e-8,
              max_iter=400)
    assert res.converged
    return cell, res


@pytest.fixture(scope="module")
def reference():
    return _solve(COORDS, A)


def test_forces_match_finite_differences(reference):
    cell, res = reference
    F = np.asarray(forces(cell, LDA(), res, grid=GRID))
    for atom, axis in [(1, 0), (0, 2)]:
        d = np.zeros_like(COORDS)
        d[atom, axis] = H
        ep = _solve(COORDS + d, A)[1].e_tot
        em = _solve(COORDS - d, A)[1].e_tot
        np.testing.assert_allclose(F[atom, axis], -(ep - em) / (2 * H),
                                   atol=2e-6)


def test_stress_matches_finite_differences(reference):
    cell, res = reference
    sigma = np.asarray(stress(cell, LDA(), res, grid=GRID))
    vol = abs(np.linalg.det(A))
    for i, j in [(0, 0), (0, 1)]:
        eps = np.zeros((3, 3))
        eps[i, j] = eps[j, i] = H                 # a symmetric strain
        e = [_solve(COORDS @ (np.eye(3) + s * eps).T,
                    A @ (np.eye(3) + s * eps).T)[1].e_tot for s in (1, -1)]
        # dE/dh = Ω (σ_ij + σ_ji)
        np.testing.assert_allclose(sigma[i, j] * (1 if i == j else 2),
                                   (e[0] - e[1]) / (2 * H) / vol, atol=1e-7)
