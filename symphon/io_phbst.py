import numpy as np
import netCDF4
from ase import Atoms
from ase.data import atomic_masses

import ase.units as units

# Use conversion factors from ASE units
# units.J is 1 Joule in eV, units._hplanck is in J*s
EV_TO_THZ = 1.0 / (units.J * units._hplanck * 1e12)


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
    """
    Alternative implementation of read_phbst_freqs_and_eigvecs using netCDF4 directly.
    """
    nc = netCDF4.Dataset(fname, 'r')

    # Read structure
    rprimd = nc.variables['primitive_vectors'][:]  # Bohr
    xred = nc.variables['reduced_atom_positions'][:]  # fractional
    znucl = nc.variables['atomic_numbers'][:]
    typat = nc.variables['atom_species'][:]  # 1-based index to znucl

    numbers = [int(znucl[t - 1]) for t in typat]
    masses = [atomic_masses[i] for i in numbers]

    cell = rprimd * units.Bohr
    atoms = Atoms(numbers=numbers,
                  masses=masses,
                  scaled_positions=xred,
                  cell=cell,
                  pbc=True)

    # Read q-points
    qpoints = nc.variables['qpoints'][:]

    # Read frequencies in eV (ABINIT writes them in eV in _PHBST.nc)
    freqs_ev = nc.variables['phfreqs'][:]
    freqs = freqs_ev * EV_TO_THZ

    # Read eigenvectors (phdispl_cart)
    # shape is (nqpt, nbranch, nbranch, 2)
    d_real = nc.variables['phdispl_cart'][:, :, :, 0]
    d_imag = nc.variables['phdispl_cart'][:, :, :, 1]
    displ_carts = d_real + 1j * d_imag

    nqpts = len(qpoints)
    nbranch = 3 * len(numbers)
    evecs = np.zeros([nqpts, nbranch, nbranch], dtype='complex128')

    for iqpt, qpt in enumerate(qpoints):
        for ibranch in range(nbranch):
            evec = displacement_cart_to_evec(displ_carts[iqpt, ibranch, :],
                                             masses,
                                             xred,
                                             qpoint=qpt,
                                             add_phase=True)
            evecs[iqpt, :, ibranch] = evec

    nc.close()
    return atoms, qpoints, freqs, evecs
