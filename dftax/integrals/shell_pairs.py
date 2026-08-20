"""Static shell-structure constants shared by the integral builders."""


from dftax.energy.gto import _CART_COMPONENTS


# Number of Cartesian components per angular momentum (l = 0..6; h/i appear
# only in auxiliary DF bases). Kept as plain ints: nothing at import time may
# build a JAX array, or the backend comes up before a multi-node run has had
# the chance to join its process group (see dftax.ks.distributed).
N_CART = {l: len(_CART_COMPONENTS[l]) for l in range(7)}

