"""Compatibility shims for third-party dependency version drift.

Phonopy 4.x moved ``get_eigenvectors`` from ``phonopy.phonon.degeneracy``
to ``phonopy.phonon.modulation`` without changing its signature, but
``spgrep-modulation`` 0.3.0 (the latest release) still imports it from the
old location.  Aliasing the symbol back restores compatibility without
patching the installed package.

Remove this module once a spgrep-modulation release carrying the fix
(upstream commit ceb5a229) is published and required.
"""

import phonopy.phonon.degeneracy as _degeneracy
import phonopy.phonon.modulation as _phonon_modulation

if not hasattr(_degeneracy, "get_eigenvectors") and hasattr(
    _phonon_modulation, "get_eigenvectors"
):
    _degeneracy.get_eigenvectors = _phonon_modulation.get_eigenvectors
