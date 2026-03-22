"""
PHBST file I/O.

Provides ``read_phbst_freqs_and_eigvecs``, which reads phonon frequencies and
eigenvectors from an ABINIT ``_PHBST.nc`` file.  A pure-netCDF4 path is always
attempted first; if it fails the abipy path is tried as a fallback (when abipy
is installed).
"""
import numpy as np
import netCDF4
from ase import Atoms
from ase.data import atomic_masses
from phonopy.structure.atoms import PhonopyAtoms

import ase.units as units

try:
    import abipy.abilab as abilab
    HAS_ABIPY = True
except ImportError:
    HAS_ABIPY = False

# Conversion factor: eV -> THz  (derived from ASE unit constants)
EV_TO_THZ = 1.0 / (units.J * units._hplanck * 1e12)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ase_atoms_to_phonopy_atoms(atoms):
    """Convert an ASE ``Atoms`` object to a ``PhonopyAtoms`` object."""
    magmoms = atoms.get_initial_magnetic_moments()
    if len(magmoms) == 0:
        magmoms = None
    return PhonopyAtoms(
        numbers=atoms.get_atomic_numbers(),
        masses=atoms.get_masses(),
        magmoms=magmoms,
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.get_cell().array,
        pbc=atoms.get_pbc(),
    )


def displacement_cart_to_evec(displ_cart, masses, scaled_positions,
                               qpoint=None, add_phase=True):
    """
    Convert a Cartesian displacement to a mass-weighted eigenvector.

    Parameters
    ----------
    displ_cart:
        Cartesian displacement ``(atom1_x, atom1_y, atom1_z, atom2_x, ...)``.
    masses:
        Atomic masses.
    scaled_positions:
        Fractional atomic positions.
    qpoint:
        Required when *add_phase* is ``True``.
    add_phase:
        Whether to multiply by the phase factor ``exp(-2πi q·r)``.
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


# ---------------------------------------------------------------------------
# Primary reader (netCDF4, no abipy dependency)
# ---------------------------------------------------------------------------

def _read_phbst_netcdf(fname):
    """Read a PHBST.nc file directly with netCDF4."""
    nc = netCDF4.Dataset(fname, 'r')

    rprimd = nc.variables['primitive_vectors'][:]   # Bohr
    xred = nc.variables['reduced_atom_positions'][:]  # fractional
    znucl = nc.variables['atomic_numbers'][:]
    typat = nc.variables['atom_species'][:]           # 1-based index into znucl

    numbers = [int(znucl[t - 1]) for t in typat]
    masses = [atomic_masses[i] for i in numbers]

    cell = rprimd * units.Bohr
    atoms = Atoms(numbers=numbers, masses=masses, scaled_positions=xred,
                  cell=cell, pbc=True)

    qpoints = nc.variables['qpoints'][:]
    freqs_ev = nc.variables['phfreqs'][:]
    freqs = freqs_ev * EV_TO_THZ

    # phdispl_cart shape: (nqpt, nbranch, nbranch, 2)  — last dim is re/im
    d_real = nc.variables['phdispl_cart'][:, :, :, 0]
    d_imag = nc.variables['phdispl_cart'][:, :, :, 1]
    displ_carts = d_real + 1j * d_imag

    nqpts = len(qpoints)
    nbranch = 3 * len(numbers)
    evecs = np.zeros([nqpts, nbranch, nbranch], dtype='complex128')

    for iqpt, qpt in enumerate(qpoints):
        for ibranch in range(nbranch):
            evec = displacement_cart_to_evec(
                displ_carts[iqpt, ibranch, :], masses, xred,
                qpoint=qpt, add_phase=True,
            )
            evecs[iqpt, :, ibranch] = evec

    nc.close()
    return atoms, qpoints, freqs, evecs


# ---------------------------------------------------------------------------
# Fallback reader (abipy)
# ---------------------------------------------------------------------------

def _read_phbst_abipy(fname):
    """Read a PHBST.nc file via abipy (fallback)."""
    ncfile = abilab.abiopen(fname)
    struct = ncfile.structure
    atoms = struct.to_ase_atoms()
    scaled_positions = struct.frac_coords

    numbers = struct.atomic_numbers
    masses = [atomic_masses[i] for i in numbers]

    phbst = ncfile.phbands
    qpoints = phbst.qpoints.frac_coords
    nqpts = len(qpoints)
    nbranch = 3 * len(numbers)
    evecs = np.zeros([nqpts, nbranch, nbranch], dtype='complex128')

    freqs_ev = phbst.phfreqs
    ev_to_thz = phbst.phfactor_ev2units("THz")
    freqs = freqs_ev * ev_to_thz

    displ_carts = phbst.phdispl_cart

    for iqpt, qpt in enumerate(qpoints):
        for ibranch in range(nbranch):
            evec = displacement_cart_to_evec(
                displ_carts[iqpt, ibranch, :], masses, scaled_positions,
                qpoint=qpt, add_phase=True,
            )
            evecs[iqpt, :, ibranch] = evec

    return atoms, qpoints, freqs, evecs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_phbst_freqs_and_eigvecs(fname):
    """
    Read phonon frequencies and eigenvectors from an ABINIT ``_PHBST.nc`` file.

    Tries the pure-netCDF4 reader first; falls back to abipy if available.

    Returns
    -------
    atoms : ase.Atoms
    qpoints : ndarray, shape (nqpt, 3)
    freqs : ndarray, shape (nqpt, nbranch)  — in THz
    evecs : ndarray, shape (nqpt, nbranch, nbranch), complex
    """
    try:
        return _read_phbst_netcdf(fname)
    except Exception as exc:
        if not HAS_ABIPY:
            raise exc

    return _read_phbst_abipy(fname)
