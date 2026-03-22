"""Utility for computing daughter space group labels using modulated supercells."""

import numpy as np
import spglib
import warnings
from symphon.symmetry_identification import get_supercell_matrix_from_qpoint

def compute_supercell_labels(phonon, qpoint, symprec=1e-5):
    """
    Compute daughter space group labels using modulated supercells (spgrep-modulation).

    For each eigenspace at the given q-point, applies standard-basis order-parameter
    directions and identifies the resulting daughter space group by building a
    modulated supercell and running spglib on it.

    Returns a flat list of (freq, daughter_sg) — one entry per OPD direction
    (i.e. one per band), sorted by frequency.  Degenerate eigenspaces of
    dimension d contribute d consecutive entries at the same frequency, one
    for each standard-basis OPD vector j=0…d-1.

    Parameters
    ----------
    phonon : Phonopy
    qpoint : tuple
    symprec : float

    Returns
    -------
    list of (float, str)  or  None on failure
    """
    try:
        from spgrep_modulation.modulation import Modulation
    except ImportError:
        return None

    try:
        qpoint_arr = np.array(qpoint)

        # Determine supercell matrix from q-point
        supercell_matrix = get_supercell_matrix_from_qpoint(qpoint_arr)

        with warnings.catch_warnings():
            # spgrep_modulation 0.3.0 emits a spurious "Inconsistent eigenvalue"
            # UserWarning when eigenvalues from irrep-projected blocks differ from
            # full dynamical matrix eigenvalues by small numerical noise.  The
            # warning is benign — the downstream computation still succeeds.
            warnings.filterwarnings(
                "ignore",
                message="Inconsistent eigenvalue",
                category=UserWarning,
                module="spgrep_modulation",
            )
            md = Modulation.with_supercell_and_symmetry_search(
                dynamical_matrix=phonon.dynamical_matrix,
                supercell_matrix=supercell_matrix,
                qpoint=qpoint_arr,
                factor=phonon.unit_conversion_factor,
                symprec=symprec,
            )

        # Build flat list: one (freq, daughter_sg) per OPD direction
        flat = []
        for i, (eigval, eigvecs, irrep) in enumerate(md.eigenspaces):
            freq = md.eigvals_to_frequencies(eigval)
            dim = eigvecs.shape[0]

            for j in range(dim):
                opd = np.zeros(dim, dtype=complex)
                opd[j] = 1.0
                amplitudes = [float(np.abs(x) * 0.1) for x in opd]
                arguments = [float(np.angle(x)) for x in opd]

                daughter = "-"
                try:
                    cell, mod = md.get_modulated_supercell_and_modulation(
                        frequency_index=i,
                        amplitudes=amplitudes,
                        arguments=arguments,
                        return_cell=True,
                    )
                    dataset = spglib.get_symmetry_dataset(
                        (cell.cell, cell.scaled_positions, cell.numbers),
                        symprec=symprec,
                    )
                    if dataset is not None:
                        daughter = f"{dataset.international}(#{dataset.number})"
                except Exception:
                    pass

                flat.append((float(freq), daughter))

        # Sort by frequency so alignment by position is robust
        flat.sort(key=lambda x: x[0])
        return flat
    except Exception:
        return None
