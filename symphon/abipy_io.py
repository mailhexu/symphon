import numpy as np
from ase.data import atomic_masses
from phonopy.structure.atoms import PhonopyAtoms

try:
    import abipy.abilab as abilab
    HAS_ABIPY = True
except ImportError:
    HAS_ABIPY = False

from .io_phbst import read_phbst_freqs_and_eigvecs as read_phbst_no_abipy
# Moments below this (in Bohr magnetons) are treated as zero. Real magnetic
# moments are O(0.1-10) muB; float round-trip noise is far below this floor.
_MAGMOM_ZERO_TOL = 1e-6


def ase_atoms_to_phonopy_atoms(atoms):
    """
    convert ase Atoms object to PhonopyAtoms object
    """
    magmoms = atoms.get_initial_magnetic_moments()
    if len(magmoms) == 0 or np.all(np.abs(magmoms) < _MAGMOM_ZERO_TOL):
        # Vanishing moments carry no information, but a non-None value makes
        # phonopy>=4 Symmetry switch to spglib's magnetic-symmetry branch,
        # which duplicates rotations (time-reversal pairs) and breaks the
        # is_primitive_cell check. Compare against a tolerance, not exact
        # zero, so round-trip noise in the moments does not trigger it.
        magmoms = None
    return PhonopyAtoms(numbers=atoms.get_atomic_numbers(),
                        masses=atoms.get_masses(),
                        magnetic_moments=magmoms,
                        scaled_positions=atoms.get_scaled_positions(),
                        cell=atoms.get_cell().array)


def displacement_cart_to_evec(displ_cart,
                              masses,
                              scaled_positions,
                              qpoint=None,
                              add_phase=True):
    """
    displ_cart: cartisien displacement. (atom1_x, atom1_y, atom1_z, atom2_x, ...)
    masses: masses of atoms.
    scaled_postions: scaled postions of atoms.
    qpoint: if phase needs to be added, qpoint must be given.
    add_phase: whether to add phase to the eigenvectors.
    """
    if add_phase and qpoint is None:
        raise ValueError('qpoint must be given if adding phase is needed')
    m = np.sqrt(np.kron(masses, [1, 1, 1]))
    evec = displ_cart * m
    if add_phase:
        phase = [
            np.exp(-2j * np.pi * np.dot(pos, qpoint))
            for pos in scaled_positions
        ]
        phase = np.kron(phase, [1, 1, 1])
        evec *= phase
        evec /= np.linalg.norm(evec)
    return evec


def read_phbst_freqs_and_eigvecs(fname):
    try:
        return read_phbst_no_abipy(fname)
    except Exception as e:
        if not HAS_ABIPY:
            raise e

    try:
        ncfile = abilab.abiopen(fname)
        try:
            struct = ncfile.structure
            atoms = ncfile.structure.to_ase_atoms()
            scaled_positions = struct.frac_coords

            numbers = struct.atomic_numbers
            masses = [atomic_masses[i] for i in numbers]

            phbst = ncfile.phbands
            qpoints = phbst.qpoints.frac_coords
            nqpts = len(qpoints)
            nbranch = 3 * len(numbers)
            evecs = np.zeros([nqpts, nbranch, nbranch], dtype='complex128')

            # Frequencies from abipy are in eV.
            # Convert to THz for phonopy (which expects THz by default).
            freqs_ev = phbst.phfreqs
            ev_to_thz = phbst.phfactor_ev2units("THz")
            freqs = freqs_ev * ev_to_thz

            displ_carts = phbst.phdispl_cart

            for iqpt, qpt in enumerate(qpoints):
                for ibranch in range(nbranch):
                    evec = displacement_cart_to_evec(displ_carts[iqpt, ibranch, :],
                                                     masses,
                                                     scaled_positions,
                                                     qpoint=qpt,
                                                     add_phase=True)
                    evecs[iqpt, :, ibranch] = evec

            return atoms, qpoints, freqs, evecs
        finally:
            close = getattr(ncfile, "close", None)
            if close is not None:
                close()
    except Exception:
        raise

