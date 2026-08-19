"""Shell-class bucketing: contraction lengths belong in the key.

``plan_pairs``/``plan_eri3c`` batch shells into classes that share one unrolled
kernel, so every member of a class is padded to the class's primitive counts.
Keying on the angular class alone made that padding global: in cc-pVDZ, sulfur
is 12s8p1d against carbon's 9s4p1d, so a single S atom took the (s,s) class
from 9x9 to 12x12 and (p,p) from 4x4 to 8x8 for *every* pair in the molecule --
and the cost is pad x pair-count, paid overwhelmingly by the pairs with no
sulfur in them. Measured on penicillin G (C16H18N2O4S) / cc-pVDZ / DF: 8x the
padded primitive work in the 3-center build, and 20.3 GiB of build scratch
against 12.4 GiB.

Keying on the contraction lengths alone is the other extreme -- no padding, but
one compiled kernel per (angular class x contraction combo), which triples the
build's compile time and buys nothing on a molecule whose heavy atoms already
share a contraction length. The planners key exactly and then merge back under
a padded-work budget (``_PAD_TOL``), so these tests pin three things: the exact
partition pads nothing, the merge stays inside its budget and is a no-op where
padding was already free, and the whole repartitioning leaves the integrals
alone.
"""

from collections import defaultdict

import numpy as np
import pytest

from dftax.basis.loader import build_basis_data
import dftax.integrals.eri3c_bucketed as bucketed
from dftax.integrals.eri3c_bucketed import (
    _PAD_TOL, _shells, eri2c_matrix_bucketed, eri3c_matrix_bucketed,
    nuclear_attraction_bucketed, overlap_kinetic_bucketed, plan_eri3c,
    plan_pairs,
)
from dftax.system.molecule import Molecule

# Methanethiol: second row (S) next to first row (C, O-like valence) and H, the
# smallest thing that reproduces the mixed-contraction-length case.
THIOL = ("S 0.000 0.000 0.000; C 0.000 0.000 1.820; "
         "H 0.960 0.000 -0.190; H -0.510 0.890 2.070; "
         "H -0.510 -0.890 2.070; H 1.030 0.000 2.070")
# P and Cl are both 12s8p1d in cc-pVDZ: an all-second-row molecule where the
# angular class is already contraction-uniform, so the finer key must not split.
PCL3 = ("P 0 0 0; Cl 1.4 1.4 0; Cl -1.4 1.4 0; Cl 0 -1.4 1.4")


@pytest.fixture
def exact_buckets(monkeypatch):
    """Plan with no merging, so the padding claim can be checked directly."""
    monkeypatch.setattr(bucketed, "_PAD_TOL", 0.0)


def _basis(atom, name="cc-pvdz"):
    mol = Molecule.from_xyz(atom, name, spherical=True)
    return build_basis_data(
        [a.split()[0] for a in atom.split(";")],
        np.asarray(mol.atom_coords()), name, spherical=True,
    )


def _nprim_by_row(basis):
    """Map each shell's starting row to its true primitive count."""
    shells, _ = _shells(basis.angular, basis.exponents)
    return {row0: npr for _l, row0, _nc, npr in shells}


def _coarsen_pairs(plan):
    """The previous planner, recovered: merge classes back onto (la, lb) and
    take the elementwise max of the primitive counts."""
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
    """Same for the 3-center plan, merging onto (la, lb, lc)."""
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
    """Sum over classes of npairs x prod(padded primitive counts): the
    primitive work the unrolled kernels actually emit."""
    return sum(len(c[4]) * c[6][0] * c[6][1] for c in classes)


def _eri3c_work(classes):
    return sum(len(c[6]) * c[9][0] * c[9][1] * c[9][2] for c in classes)


def test_pair_classes_do_not_pad_primitive_counts(exact_buckets):
    """With merging off, no shell in a pair class is padded: every member
    carries exactly the class's primitive counts. This is the property the old
    angular-only key violated, and the floor the merge budget is measured
    against."""
    basis = _basis(THIOL)
    npr = _nprim_by_row(basis)
    for la, lb, _anga, _angb, rows_a, rows_b, (npa, npb) in plan_pairs(basis)[1]:
        assert {npr[r] for r in rows_a} == {npa}, (la, lb)
        assert {npr[r] for r in rows_b} == {npb}, (la, lb)


def test_eri3c_classes_do_not_pad_primitive_counts(exact_buckets):
    """Same for the 3-center plan, over both the orbital and auxiliary legs."""
    basis, aux = _basis(THIOL), _basis(THIOL, "def2-universal-jkfit")
    npr, npr_aux = _nprim_by_row(basis), _nprim_by_row(aux)
    for c in plan_eri3c(basis, aux)[4]:
        rows_a, rows_b, rows_c, (npa, npb, npc) = c[6], c[7], c[8], c[9]
        assert {npr[r] for r in rows_a} == {npa}
        assert {npr[r] for r in rows_b} == {npb}
        assert {npr_aux[r] for r in rows_c} == {npc}


def test_sulfur_does_not_resize_the_first_row_kernels():
    """The regression, quantified, at the shipped merge budget. Bucketing
    methanethiol on the angular class alone pads the whole molecule to sulfur's
    contraction lengths; the shipped plan cuts the emitted primitive work
    several-fold, which is what the build's peak scratch tracks."""
    basis, aux = _basis(THIOL), _basis(THIOL, "def2-universal-jkfit")
    pairs, three = plan_pairs(basis), plan_eri3c(basis, aux)
    old_pairs, old_three = _coarsen_pairs(pairs), _coarsen_eri3c(three)

    # More classes, each sized nearer to its own members -- the trade.
    assert len(pairs[1]) > len(old_pairs[1])
    assert len(three[4]) > len(old_three[4])
    assert _pair_work(pairs[1]) * 2 < _pair_work(old_pairs[1])
    assert _eri3c_work(three[4]) * 3 < _eri3c_work(old_three[4])


def test_merge_stays_inside_its_padding_budget(monkeypatch):
    """Merging never carries the plan more than ``_PAD_TOL`` above the exact
    partition's padded work -- the guarantee that lets the budget be raised for
    compile time without silently reintroducing the sulfur blow-up."""
    basis, aux = _basis(THIOL), _basis(THIOL, "def2-universal-jkfit")
    shipped = (_pair_work(plan_pairs(basis)[1]),
               _eri3c_work(plan_eri3c(basis, aux)[4]))
    monkeypatch.setattr(bucketed, "_PAD_TOL", 0.0)
    exact = (_pair_work(plan_pairs(basis)[1]),
             _eri3c_work(plan_eri3c(basis, aux)[4]))
    for got, floor in zip(shipped, exact):
        assert floor <= got <= floor * (1.0 + _PAD_TOL)


def test_uniform_contraction_lengths_leave_the_partition_alone():
    """P and Cl are both 12s8p1d in cc-pVDZ, so PCl3's angular classes are
    already contraction-uniform: the finer key never splits them, and the merge
    has nothing to put back. Same classes, same work as the angular-only plan.
    The fix costs kernels only where padding was real."""
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
    """Splitting a class only repartitions the same shell pairs into more
    kernels, so the assembled matrices are unchanged. Not bit-identical:
    padded primitives contribute zero but still shift the summation order
    inside a kernel, which lands at the 1e-14 level.
    """
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
    """Same claim for the 3-center build, whose class count grows fastest
    (the l-triple times the contraction triple).

    The orbital basis stands in for the auxiliary one: this is a statement
    about partitioning, not about fitting, and a real JK-fitting set is a few
    hundred extra shells of compile for no extra coverage.
    """
    basis = _basis(THIOL)
    three = plan_eri3c(basis, basis)
    _agrees(lambda p: eri3c_matrix_bucketed(basis, basis, plan=p),
            three, _coarsen_eri3c(three))


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
            assert lo == prev_c and slo == prev_s      # contiguous cover
            assert lo in starts                        # never cuts a shell
            assert hi > lo and shi > slo
            prev_c, prev_s = hi, shi
        assert prev_c == aux.centers.shape[0]
        assert prev_s == aux.cart2sph.shape[1]
