# Implicit differentiation (CPHF response)

Energy gradients (forces, `∂E/∂Z`) need no implicit differentiation: at the variational
minimum `∂E/∂C = 0`, so the [forces](forces.md) come straight from the explicit geometry
derivative. What *does* need the orbital response is the derivative of the **converged
density** with respect to a parameter, and hence first-order response properties.

`implicit_density(ks)` is the converged density `P*` as a function of the assembled
`KS` (restricted / closed-shell only), made differentiable by implicit differentiation
of the SCF fixed point:

- the **forward** runs the ordinary SCF under `stop_gradient`;
- the **backward** solves the response (CPHF) equation `(I − ∂g/∂P)ᵀ w = P̄` matrix-free
  (GMRES), the one-step Jacobian coming from `jax.vjp` of a single SCF step (reusing the
  autodiff Fock), with the eigendecomposition's unstable derivative replaced by a stable
  occ-virt projector response.

Because the backward differentiates the assembled functional, all Pulay / basis
derivatives are supplied by the engine's own autodiff. Backward memory is independent
of the SCF iteration count.

## Analytic polarizability

The headline use is the exact coupled-perturbed-KS polarizability, a single
`jax.jacobian` of the dipole through the converged density, no field stepping:

```python
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
from dftax import Molecule, polarizability
from dftax.energy.xc import PBE

mol = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "sto-3g")
a_analytic = np.asarray(polarizability(mol, PBE(), method="analytic"))
a_fd       = np.asarray(polarizability(mol, PBE(), method="fd"))
print(np.max(np.abs(a_analytic - a_fd)))     # ~2e-6  (analytic CPHF == finite field)
```

Validation: the energy gradient *through* `implicit_density` reproduces the analytic
forces to ~7e-9, and the analytic polarizability matches the finite-field tensor to
~2e-6.

## Analytic Hessian

The analytic geometry Hessian is available: `hessian(mol, xc,
method="analytic")` assembles the exact orbital-rotation Schur complement
`H = E_RR − E_Rκ (E_κκ)⁻¹ E_κR` at one tightly converged (Newton-polished)
reference, one CG response solve per column, so 3N response solves replace
the finite-difference path's 6N SCF solves. RKS and UKS are supported, on
the materialized Coulomb backends by default; an explicit `df(chunk=<int>)`
streams the response through the forces backend's frozen exchange with the
traced rotated occupieds, so the 3-center tensor is never materialized.
`method="fd"` (the default) remains the finite difference of the analytic
forces (water/sto-3g frequencies match PySCF to <5 cm⁻¹).

A historical note: this was long blocked on a NaN in the second geometric
derivative of the energy at fixed density; the culprit turned out to be the
retired pre-0.4.0 flat integral paths, and the shell-class bucketed engine
is cleanly twice-differentiable. On density-fitted surfaces, validate
second derivatives against exact-backend parity or large-step energy finite
differences, never small-step FD: the FD legs' RI-metric noise (~2e-10 Ha
per solve) amplifies as 1/h² and swamps the comparison.
