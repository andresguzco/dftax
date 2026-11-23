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
