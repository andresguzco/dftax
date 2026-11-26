from collections import defaultdict

import numpy as np
import pytest

from dftax.basis.loader import build_basis_data
import dftax.integrals.eri3c_bucketed as bucketed
from dftax.integrals.eri3c_bucketed import (
    _PAD_TOL, _shells, eri2c_matrix_bucketed, eri3c_matrix_bucketed,
    expand_eri3c, nuclear_attraction_bucketed, overlap_kinetic_bucketed,
    plan_eri3c, plan_pairs,
)
from dftax.system.molecule import Molecule

THIOL = ("S 0.000 0.000 0.000; C 0.000 0.000 1.820; "
         "H 0.960 0.000 -0.190; H -0.510 0.890 2.070; "
         "H -0.510 -0.890 2.070; H 1.030 0.000 2.070")
PCL3 = ("P 0 0 0; Cl 1.4 1.4 0; Cl -1.4 1.4 0; Cl 0 -1.4 1.4")


@pytest.fixture
def exact_buckets(monkeypatch):
    monkeypatch.setattr(bucketed, "_PAD_TOL", 0.0)


def _basis(atom, name="cc-pvdz"):
    mol = Molecule.from_xyz(atom, name, spherical=True)
    return build_basis_data(
        [a.split()[0] for a in atom.split(";")],
        np.asarray(mol.atom_coords()), name, spherical=True,
    )


def _nprim_by_row(basis):
    shells, _ = _shells(basis.angular, basis.exponents)
    return {row0: npr for _l, row0, _nc, npr in shells}


def _coarsen_pairs(plan):
    nao, classes = plan
    m = defaultdict(lambda: [None, None, [], [], (0, 0)])
    for la, lb, anga, angb, ra, rb, npr in classes:
        e = m[(la, lb)]
        e[0], e[1] = anga, angb
        e[2].extend(ra)
        e[3].extend(rb)
        e[4] = (max(e[4][0], npr[0]), max(e[4][1], npr[1]))
    return (nao, tuple(
        (la, lb, e[0], e[1], tuple(e[2]), tuple(e[3]), e[4])
        for (la, lb), e in sorted(m.items())))


def _coarsen_eri3c(plan):
    nao, naux, nsph, nauxsph, classes = plan
    m = defaultdict(
        lambda: [None, None, None, [], [], [], (0, 0, 0), [], [], []])
    for c in classes:
        la, lb, lc, anga, angb, angc, ra, rb, rc, npr, sa, sb, sc = c
        e = m[(la, lb, lc)]
        e[0], e[1], e[2] = anga, angb, angc
        e[3].extend(ra)
        e[4].extend(rb)
        e[5].extend(rc)
        e[6] = tuple(max(x, y) for x, y in zip(e[6], npr))
        e[7].extend(sa)
        e[8].extend(sb)
        e[9].extend(sc)
    return (nao, naux, nsph, nauxsph, tuple(
        (la, lb, lc, e[0], e[1], e[2], tuple(e[3]), tuple(e[4]), tuple(e[5]),
         e[6], tuple(e[7]), tuple(e[8]), tuple(e[9]))
        for (la, lb, lc), e in sorted(m.items())))


def _pair_work(classes):
    return sum(len(c[4]) * c[6][0] * c[6][1] for c in classes)


def _eri3c_work(classes):
    return sum(len(c[6]) * c[9][0] * c[9][1] * c[9][2] for c in classes)


def test_pair_classes_do_not_pad_primitive_counts(exact_buckets):
    basis = _basis(THIOL)
    npr = _nprim_by_row(basis)
    for la, lb, _anga, _angb, rows_a, rows_b, (npa, npb) in plan_pairs(basis)[1]:
        assert {npr[r] for r in rows_a} == {npa}, (la, lb)
        assert {npr[r] for r in rows_b} == {npb}, (la, lb)


def test_eri3c_classes_do_not_pad_primitive_counts(exact_buckets):
    basis, aux = _basis(THIOL), _basis(THIOL, "def2-universal-jkfit")
    npr, npr_aux = _nprim_by_row(basis), _nprim_by_row(aux)
    for c in expand_eri3c(plan_eri3c(basis, aux))[4]:
        rows_a, rows_b, rows_c, (npa, npb, npc) = c[6], c[7], c[8], c[9]
        assert {npr[r] for r in rows_a} == {npa}
        assert {npr[r] for r in rows_b} == {npb}
        assert {npr_aux[r] for r in rows_c} == {npc}


def test_sulfur_does_not_resize_the_first_row_kernels():
    basis, aux = _basis(THIOL), _basis(THIOL, "def2-universal-jkfit")
    pairs, three = plan_pairs(basis), expand_eri3c(plan_eri3c(basis, aux))
    old_pairs, old_three = _coarsen_pairs(pairs), _coarsen_eri3c(three)

    assert len(pairs[1]) > len(old_pairs[1])
    assert len(three[4]) > len(old_three[4])
    assert _pair_work(pairs[1]) * 2 < _pair_work(old_pairs[1])
    assert _eri3c_work(three[4]) * 3 < _eri3c_work(old_three[4])


def test_merge_stays_inside_its_padding_budget(monkeypatch):
    basis, aux = _basis(THIOL), _basis(THIOL, "def2-universal-jkfit")
    shipped = (_pair_work(plan_pairs(basis)[1]),
               _eri3c_work(expand_eri3c(plan_eri3c(basis, aux))[4]))
    monkeypatch.setattr(bucketed, "_PAD_TOL", 0.0)
    exact = (_pair_work(plan_pairs(basis)[1]),
             _eri3c_work(expand_eri3c(plan_eri3c(basis, aux))[4]))
    for got, floor in zip(shipped, exact):
        assert floor <= got <= floor * (1.0 + _PAD_TOL)


def test_uniform_contraction_lengths_leave_the_partition_alone():
    basis = _basis(PCL3)
    pairs = plan_pairs(basis)
    assert pairs == _coarsen_pairs(pairs)


def _agrees(build, new, old, tol=1e-11):
    got, ref = build(new), build(old)
    got = got if isinstance(got, tuple) else (got,)
    ref = ref if isinstance(ref, tuple) else (ref,)
    for x, y in zip(got, ref):
        assert float(np.abs(np.asarray(x) - np.asarray(y)).max()) < tol


@pytest.mark.float64
@pytest.mark.slow
def test_finer_pair_buckets_leave_the_integrals_alone():
    basis = _basis(THIOL)
    mol = Molecule.from_xyz(THIOL, "cc-pvdz", spherical=True)
    xyz = np.asarray(mol.atom_coords())
    charges = np.asarray(mol.atom_charges(), dtype=float)
    pairs = plan_pairs(basis)
    coarse = _coarsen_pairs(pairs)
    _agrees(lambda p: overlap_kinetic_bucketed(basis, plan=p), pairs, coarse)
    _agrees(lambda p: nuclear_attraction_bucketed(basis, xyz, charges, plan=p),
            pairs, coarse)
    _agrees(lambda p: eri2c_matrix_bucketed(basis, plan=p), pairs, coarse)


@pytest.mark.float64
@pytest.mark.slow
def test_finer_eri3c_buckets_leave_the_integrals_alone():
    basis = _basis(THIOL)
    three = plan_eri3c(basis, basis)
    _agrees(lambda p: eri3c_matrix_bucketed(basis, basis, plan=p),
            three, _coarsen_eri3c(expand_eri3c(three)))


COORDS3 = np.array([[0.0, 0.0, 0.0], [1.5, 0.1, 0.0], [-0.4, 1.4, 0.2]])


@pytest.mark.float64
class TestAuxSlabs:
    def test_slab_concat_matches_full_tensor(self):
        from dftax.integrals.eri3c_bucketed import plan_aux_slabs, slice_aux

        basis = build_basis_data(["O", "H", "H"], COORDS3, "sto-3g",
                                 spherical=True)
        aux = build_basis_data(["O", "H", "H"], COORDS3, "def2-svp",
                               spherical=True)
        full = np.asarray(eri3c_matrix_bucketed(basis, aux))
        slabs = plan_aux_slabs(basis, aux, 6)
        assert len(slabs) > 1
        cat = np.concatenate([
            np.asarray(eri3c_matrix_bucketed(
                basis, slice_aux(aux, lo, hi, slo, shi), plan=plan))
            for (lo, hi, slo, shi, plan) in slabs
        ], axis=2)
        assert cat.shape == full.shape
        assert np.abs(cat - full).max() < 1e-13

    def test_slabs_are_shell_aligned_and_contiguous(self):
        from dftax.integrals.eri3c_bucketed import plan_aux_slabs

        aux = build_basis_data(["O", "H", "H"], COORDS3, "def2-svp",
                               spherical=True)
        basis = build_basis_data(["O", "H", "H"], COORDS3, "sto-3g",
                                 spherical=True)
        shells, _ = _shells(aux.angular, aux.exponents)
        starts = {r0 for (_, r0, _, _) in shells}
        slabs = plan_aux_slabs(basis, aux, 5)
        prev_c = prev_s = 0
        for (lo, hi, slo, shi, _plan) in slabs:
            assert lo == prev_c and slo == prev_s
            assert lo in starts
            assert hi > lo and shi > slo
            prev_c, prev_s = hi, shi
        assert prev_c == aux.centers.shape[0]
        assert prev_s == aux.cart2sph.shape[1]


@pytest.mark.float64
class TestBucketedEri4c:
    def test_matches_flat_cartesian(self):
        import jax.numpy as jnp

        from dftax.integrals.eri4c import (
            _eri4c_matrix_flat, _unique_quartets, eri4c_matrix,
        )

        b = build_basis_data(["O", "H", "H"], COORDS3, "sto-3g")
        qi, qj, qk, ql, qof = _unique_quartets(b.centers.shape[0])
        quart = jnp.stack([jnp.asarray(x) for x in (qi, qj, qk, ql)], axis=1)
        ref = np.asarray(_eri4c_matrix_flat(b, quartets=quart, qof=qof))
        new = np.asarray(eri4c_matrix(b))
        assert np.abs(new - ref).max() < 1e-12

    @pytest.mark.slow
    def test_matches_flat_spherical_d_shells_lr(self):
        import jax.numpy as jnp

        from dftax.integrals.eri4c import (
            _eri4c_matrix_flat, _unique_quartets, eri4c_matrix,
        )

        b = build_basis_data(["O", "H"], COORDS3[:2], "def2-svp",
                             spherical=True)
        qi, qj, qk, ql, qof = _unique_quartets(b.centers.shape[0])
        quart = jnp.stack([jnp.asarray(x) for x in (qi, qj, qk, ql)], axis=1)
        for omega in (None, 0.33):
            ref = np.asarray(
                _eri4c_matrix_flat(b, quartets=quart, qof=qof, omega=omega)
            )
            new = np.asarray(eri4c_matrix(b, omega=omega))
            assert np.abs(new - ref).max() < 1e-12


def test_factored_plan_holds_every_triple_once():
    basis, aux = _basis(THIOL), _basis(THIOL, "def2-universal-jkfit")
    bra, _ = _shells(basis.angular, basis.exponents)
    shells_aux, _ = _shells(aux.angular, aux.exponents)
    seen = []
    for c in expand_eri3c(plan_eri3c(basis, aux))[4]:
        seen.extend(zip((min(a, b) for a, b in zip(c[6], c[7])),
                        (max(a, b) for a, b in zip(c[6], c[7])), c[8]))
    want = [(bra[i][1], bra[j][1], k[1]) for i in range(len(bra))
            for j in range(i, len(bra)) for k in shells_aux]
    assert sorted(seen) == sorted(want)


def test_factored_plan_is_linear_in_pairs_and_aux_shells():
    basis, aux = _basis(THIOL), _basis(THIOL, "def2-universal-jkfit")
    plan = plan_eri3c(basis, aux)

    def rows(c):
        if len(c) == 8:
            return sum(4 * len(p[0]) + 2 * len(a[0]) for p, a in c[6])
        return 6 * len(c[6])

    factored = sum(rows(c) for c in plan[4])
    expanded = sum(rows(c) for c in expand_eri3c(plan)[4])
    assert factored * 5 < expanded


@pytest.mark.float64
def test_factored_build_is_the_expanded_build():
    basis = _basis(THIOL, "sto-3g")
    aux = _basis(THIOL, "6-31g")
    plan = plan_eri3c(basis, aux)
    np.testing.assert_array_equal(
        eri3c_matrix_bucketed(basis, aux, plan=plan),
        eri3c_matrix_bucketed(basis, aux, plan=expand_eri3c(plan)))
