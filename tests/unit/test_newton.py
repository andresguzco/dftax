"""Second-order SCF (newton): parity, quadratic cleanup, hard-case roles."""


import pytest

from dftax import KS, Molecule, becke, newton, scf
from dftax.energy.xc import PBE

WATER = "O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50"


@pytest.mark.float64
def test_newton_matches_scf_on_easy_case():
    """Same fixed point as DIIS, in fewer iterations (quadratic tail)."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    r0 = scf(ks)
    r1 = newton(ks)
    assert r0.converged and r1.converged
    assert abs(r1.e_tot - r0.e_tot) < 1e-7
    assert r1.n_iter <= r0.n_iter


@pytest.mark.float64
def test_newton_warm_start_is_quadratic_cleanup():
    """From a converged density the Newton step count is O(1), on an open
    shell (the O2 triplet). Not the Fe atom it used to be: there plain DIIS
    reaches different stationary points from different guesses and, on CPU,
    none inside 200 iterations, so the test stood on its reference solver
    rather than on Newton."""
    mol = Molecule.from_xyz("O 0 0 0; O 0 0 1.21", "sto-3g", spin=2)
    ks = KS(mol, PBE(), grid=becke(35, 50), spin=2)
    r0 = scf(ks)
    r1 = newton(ks, guess=r0.P)
    assert r0.converged and r1.converged
    assert r1.n_iter <= 3
    assert abs(r1.e_tot - r0.e_tot) < 1e-7


@pytest.mark.float64
def test_newton_cold_start_stretched_bond():
    """Stretched N2 converges cold in a handful of Newton steps."""
    mol = Molecule.from_xyz("N 0 0 0; N 0 0 2.0", "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    r0 = scf(ks)
    r1 = newton(ks)
    assert r0.converged and r1.converged
    assert r1.n_iter <= 8
    assert abs(r1.e_tot - r0.e_tot) < 1e-7


@pytest.mark.float64
def test_newton_escapes_indefinite_hessian():
    """A strongly stretched N2 (2.5 A) sits at an ill-conditioned/indefinite
    point where plain-CG Newton stalls: the decrease-only trust region cannot
    make progress from the direction a positive-definite CG solve returns, so
    it never converges. The Steihaug-Toint truncated-CG step follows negative
    curvature to the trust boundary and converges to a critical point (which
    restricted-symmetry critical point of a stretched bond is basin-dependent;
    the claim here is anti-stall, not global optimality -- easy cases with a
    clear minimum match DIIS, see test_newton_matches_scf_on_easy_case).
    """
    mol = Molecule.from_xyz("N 0 0 0; N 0 0 2.5", "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    # Was known flaky: with XLA's GEMM autotuning on, this took 3 steps on some
    # runs and failed to converge in 120 on others. The cause was the trust
    # region's strict-decrease test: near convergence a Newton step lowers E
    # by about as much as the rounding the autotuner's kernel choice moves,
    # so steps were rejected at random and every rejection halved the radius
    # until the solve stalled at the answer. newton() now accepts a step
    # within the energy's noise floor and never asks the energy test for less
    # than it (newton._noise), the same rule as gdm and scf.
    r1 = newton(ks, max_iter=120)
    assert r1.converged                                # no longer stalls
    assert r1.n_iter <= 60
    assert -108.0 < float(r1.e_tot) < -106.0           # a physical N2 solution


@pytest.mark.float64
def test_newton_reaches_tight_tolerances_directly():
    """The coarse-grid tight-tolerance case from the DF conditioning study:
    Newton drives the orbital gradient to g_tol=1e-7 in a handful of steps.
    (Whether DIIS from a core start grinds against its noise floor here is a
    knife edge that last-digit integral changes flip, so it is not
    asserted.)"""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    # g_tol=1e-7, not 1e-9. This case sits on the achievable-gradient floor of
    # a becke(35, 50) grid, and 1e-9 is below it: with XLA's GEMM autotuning
    # disabled, so that kernel selection is reproducible, Newton reaches 1e-7
    # in 4 steps, 1e-8 in 40, and does not reach 1e-9 inside the 64-step budget
    # at all -- while every one of those lands the same energy to 1e-9 Ha. The
    # test used to ask for 1e-9 and passed only on the autotuning draws that
    # happened to fall the right way, which is why it failed roughly one run in
    # three. Asking for a tolerance the grid can actually resolve makes it
    # deterministic without pinning the platform's kernel choice.
    r1 = newton(ks, g_tol=1e-7, e_tol=1e-12)
    assert r1.converged
    # A budget, not a pin: the step count still moves with last-digit changes
    # in the integrals (on CPU, 7 vs 43 across two Boys tables agreeing to
    # 2.5e-14). The claim is that Newton closes a tolerance DIIS grinds
    # against, not that it takes exactly four steps.
    assert r1.n_iter <= 50
