"""Fail if any public symbol (dftax.__all__) is missing from the rendered docs.

Optax enforces the same invariant with a custom Sphinx extension; here a name
counts as documented when some docs page carries a ``::: <path>`` mkdocstrings
directive whose final segment is the symbol (re-exports are documented under
their defining path, e.g. ``::: dftax.ks.scf.scf`` covers ``dftax.scf``).
Run from the repo root: ``python scripts/docs/check_coverage.py``.
"""

from __future__ import annotations

import pathlib
import re
import sys

import dftax

# Not API: metadata and convenience re-exports documented as part of their owner.
SKIP = {"__version__"}

directives = set()
for md in pathlib.Path("docs").rglob("*.md"):
    for m in re.finditer(r"^::: ([\w.]+)", md.read_text(), re.MULTILINE):
        directives.add(m.group(1).rsplit(".", 1)[-1])

missing = [n for n in dftax.__all__ if n not in SKIP and n not in directives]
if missing:
    print("Public symbols missing from the docs (no `::: path` directive):")
    for n in missing:
        print(f"  - {n}")
    sys.exit(1)
print(f"doc coverage OK: {len(dftax.__all__) - len(SKIP)} public symbols documented")
