import numpy as np
from ase.data import atomic_masses
import abipy.abilab as abilab
from phonopy.structure.atoms import PhonopyAtoms


def ase_atoms_to_phonopy_atoms(atoms):
    """
    convert ase Atoms object to PhonopyAtoms object
    """
    magmoms = atoms.get_initial_magnetic_moments()
    if len(magmoms) == 0:
        magmoms = None
    return PhonopyAtoms(numbers=atoms.get_atomic_numbers(),
                        masses=atoms.get_masses(),
                        magmoms=magmoms,
                        scaled_positions=atoms.get_scaled_positions(),
                        cell=atoms.get_cell().array,
                        pbc=atoms.get_pbc())


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
    ncfile = abilab.abiopen(fname)
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

    # Frequencies from abipy are in Hartree atomic units.
    # Convert to THz for phonopy (which expects THz by default).
    freqs_ha = phbst.phfreqs
    ha_to_thz = phbst.phfactor_ev2units("THz")
    freqs = freqs_ha * ha_to_thz
    
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
