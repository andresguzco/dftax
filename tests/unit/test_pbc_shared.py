import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax.pbc import (
    Cell, ewald_energy, gvectors, hartree_energy, load_basis_shells,
    load_potential, local_potential, plan_ewald, uniform_grid, vloc_G,
)
from dftax.pbc.gth import local_parameters

A = np.array([[9.0, 0.0, 0.0], [2.0, 8.5, 0.0], [1.0, -1.5, 10.0]])
ATOMS = "Si 0.5 0.4 0.3; O 3.4 1.1 0.9; O 1.0 4.2 5.6"
MESH = (9, 10, 12)


@pytest.fixture(scope="module")
def cell():
    return Cell.from_xyz(ATOMS, A, "gth-szv", "gth-pade", unit="bohr")


@pytest.fixture(scope="module")
def pcell():
    from pyscf.pbc import gto

    c = gto.Cell()
    c.atom, c.a, c.unit = ATOMS, A, "B"
    c.basis, c.pseudo = "gth-szv", "gth-pade"
    c.mesh = list(MESH)
    c.precision = 1e-12
    c.verbose = 0
    c.build()
    return c


def test_lattice_matches_pyscf(cell, pcell):
    G = gvectors(jnp.asarray(cell.a), MESH)
    np.testing.assert_allclose(G, pcell.get_Gv(list(MESH)), atol=1e-13)
    r, w = uniform_grid(jnp.asarray(cell.a), MESH)
    ref = pcell.get_uniform_grids(list(MESH), wrap_around=False)
    np.testing.assert_allclose(r, ref, atol=1e-13)
    np.testing.assert_allclose(w, pcell.vol / len(ref), rtol=1e-14)


def test_ewald_matches_pyscf(cell, pcell):
    coords = jnp.asarray(cell.atom_coords())
    Z = jnp.asarray(cell.atom_charges())
    a = jnp.asarray(cell.a)
    np.testing.assert_allclose(ewald_energy(coords, Z, a), pcell.ewald(),
                               atol=1e-10)

    plan = plan_ewald(cell.a, cell.atom_coords())
    g = jax.grad(lambda x: ewald_energy(x, Z, a, plan))(coords)
    assert np.all(np.isfinite(g))
    np.testing.assert_allclose(np.sum(g, axis=0), 0.0, atol=1e-10)
    h = 1e-4
    d = jnp.zeros_like(coords).at[1, 2].set(h)
    fd = (ewald_energy(coords + d, Z, a, plan)
          - ewald_energy(coords - d, Z, a, plan)) / (2 * h)
    np.testing.assert_allclose(g[1, 2], fd, atol=1e-7)
    ga = jax.grad(lambda x: ewald_energy(coords, Z, x, plan))(a)
    assert np.all(np.isfinite(ga))


def test_gth_tables_match_pyscf():
    from pyscf.pbc.gto import basis as pbasis, pseudo as ppseudo

    for sym in ("Si", "O"):
        pot = load_potential("gth-pade", sym)
        ref = ppseudo.load("gth-pade", sym)
        assert pot.zion == sum(ref[0])
        assert pot.rloc == ref[1]
        assert list(pot.cloc[:ref[2]]) == list(ref[3])
        assert len(pot.projectors) == ref[4]
        for (r_l, h), (r_ref, n_ref, h_ref) in zip(pot.projectors, ref[5:]):
            assert r_l == r_ref and h.shape == (n_ref, n_ref)
            np.testing.assert_array_equal(h, np.asarray(h_ref).reshape(n_ref, n_ref))

        shells = load_basis_shells("gth-dzvp", sym)
        ref = pbasis.load("gth-dzvp", sym)
        assert [s["angular_momentum"][0] for s in shells] == [b[0] for b in ref]
        for s, b in zip(shells, ref):
            rows = np.asarray(b[1:])
            np.testing.assert_array_equal(s["exponents"], rows[:, 0])
            np.testing.assert_array_equal(np.asarray(s["coefficients"]).T, rows[:, 1:])


def test_local_potential_matches_pyscf(cell, pcell):
    from pyscf.pbc import tools
    from pyscf.pbc.gto import pseudo as ppseudo

    Gv = pcell.get_Gv(list(MESH))
    ref_G = ppseudo.get_vlocG(pcell, Gv)
    zion, rloc, cloc = map(jnp.asarray, local_parameters(cell.potentials))
    G2 = jnp.sum(jnp.asarray(Gv) ** 2, axis=-1)
    ours = jax.vmap(lambda z, r, c: vloc_G(G2, z, r, c))(zion, rloc, cloc)
    np.testing.assert_allclose(ours, -ref_G, rtol=1e-12, atol=1e-12)

    ref_R = tools.ifft(-np.einsum("ij,ij->j", pcell.get_SI(mesh=list(MESH)), ref_G),
                       list(MESH)).real
    v = local_potential(jnp.asarray(cell.atom_coords()), jnp.asarray(cell.a),
                        MESH, zion, rloc, cloc)
    np.testing.assert_allclose(v, ref_R, rtol=1e-11, atol=1e-12)


def test_hartree_energy_matches_pyscf(cell, pcell):
    from pyscf.pbc import tools

    rho = np.random.default_rng(0).random(int(np.prod(MESH)))
    v = tools.ifft(tools.get_coulG(pcell, mesh=list(MESH))
                   * tools.fft(rho, list(MESH)), list(MESH)).real
    ref = 0.5 * pcell.vol / rho.size * np.dot(rho, v)
    np.testing.assert_allclose(
        hartree_energy(jnp.asarray(rho), jnp.asarray(cell.a), MESH), ref,
        rtol=1e-12)
