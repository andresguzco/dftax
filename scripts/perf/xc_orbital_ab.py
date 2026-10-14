"""XC through the occupied orbitals against XC through the density matrix.

    source scripts/perf/env.sh
    python scripts/perf/xc_orbital_ab.py --mol cubane bicyclo222octane

On the materialized grid (:class:`~dftax.ks.terms.GridXC`), a direct
minimizer needs the XC energy and its derivative with respect to the occupied
coefficients. Today that goes through the density matrix: ``rho`` from
``ao P ao`` (``ng nao^2``) and the potential matrix ``V = dE/dP`` from the
backward (``ng nao^2`` again), then ``2 V C``. Through the orbitals,
``psi = ao C`` and ``grad psi = dao C`` cost ``4 ng nao nocc`` forward and as
much backward, and the derivative with respect to ``C`` comes out directly.
This prices both on the same KS, closed-shell GGA, and checks they agree.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from profile_terms import read_xyz  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mol", nargs="*", default=["cubane"])
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from dftax import KS, Molecule, becke, minao
    from dftax.energy.grid import xc_energy
    from dftax.energy.xc import PBE
    from dftax.ks.guess import density_from_guess
    from dftax.ks.scf import canonical_orthonormalizer

    def timed(fn, *a):
        jax.block_until_ready(fn(*a))
        walls = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*a))
            walls.append(time.perf_counter() - t0)
        return statistics.median(walls) * 1e3

    for name in args.mol:
        mol = Molecule.from_xyz(read_xyz(name), args.basis, spherical=True)
        ks = KS(mol, PBE(), grid=becke(chunk=None))
        xt = ks.xc_term
        X = canonical_orthonormalizer(ks.S)
        P0 = density_from_guess(ks, minao(), X)
        _, F = jax.jit(ks.energy_and_fock)(P0)
        _, Cp = jnp.linalg.eigh(X.T @ F[0] @ X)
        nocc = ks.nocc[0]
        C = (X @ Cp)[:, :nocc] * jnp.sqrt(2.0)          # P = C C^T

        @jax.jit
        def via_density(C):
            e, V = xt.energy_and_potential((C @ C.T)[None])
            return e, 2.0 * V[0] @ C                    # dE/dC

        def e_orb(C):
            psi = xt.ao @ C                             # (ng, nocc)
            dpsi = jnp.einsum("gmx,mi->gix", xt.dao, C)
            rho = jnp.sum(psi * psi, axis=1)
            grad = 2.0 * jnp.einsum("gi,gix->gx", psi, dpsi)
            return xc_energy(xt.xc, rho, xt.weights, grad_rho=grad)

        via_orbitals = jax.jit(jax.value_and_grad(e_orb))
        e1, g1 = via_density(C)
        e2, g2 = via_orbitals(C)
        t1 = timed(via_density, C)
        t2 = timed(via_orbitals, C)
        print(f"{name:17s} ng {xt.weights.shape[0]} nao {C.shape[0]} nocc "
              f"{nocc}: density-matrix {t1:6.2f} ms  orbitals {t2:6.2f} ms  "
              f"x{t1 / t2:.2f}  |dE| {abs(float(e1 - e2)):.1e}  "
              f"|dg| {float(jnp.max(jnp.abs(g1 - g2))):.1e}", flush=True)


if __name__ == "__main__":
    main()
