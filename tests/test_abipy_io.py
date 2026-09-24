"""Tests for ASE and phonopy structure conversion."""

import numpy as np
from ase import Atoms

from symphon.abipy_io import ase_atoms_to_phonopy_atoms


def test_ase_atoms_to_phonopy_atoms_preserves_magnetic_moments():
    atoms = Atoms(
        "Fe2",
        cell=np.eye(3) * 3.0,
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        pbc=True,
    )
    atoms.set_initial_magnetic_moments([2.0, -2.0])

    converted = ase_atoms_to_phonopy_atoms(atoms)

    np.testing.assert_array_equal(converted.numbers, atoms.get_atomic_numbers())
    np.testing.assert_allclose(
        converted.magnetic_moments,
        atoms.get_initial_magnetic_moments(),
    )


def test_ase_atoms_to_phonopy_atoms_drops_zero_magnetic_moments():
    """Non-magnetic structures must not trigger phonopy's magnetic-symmetry branch.

    ASE returns an all-zeros array for non-magnetic Atoms. Forwarding it as
    ``magnetic_moments=[0, ...]`` makes phonopy>=4 ``Symmetry`` return a
    ``SpglibMagneticDataset`` whose rotations come in time-reversal pairs,
    which breaks ``is_primitive_cell`` and raises the spurious
    "Non-primitive cell" RuntimeError.
    """
    from phonopy.structure.cells import is_primitive_cell
    from phonopy.structure.symmetry import Symmetry

    atoms = Atoms(
        "SiO",
        cell=[[4.0, 0, 0], [0, 5.0, 0], [0.6, 0, 7.0]],
        scaled_positions=[[0.0, 0.0, 0.0], [0.31, 0.5, 0.27]],
        pbc=True,
    )
    assert np.all(atoms.get_initial_magnetic_moments() == 0.0)

    converted = ase_atoms_to_phonopy_atoms(atoms)

    assert converted.magnetic_moments is None
    dataset = Symmetry(converted, symprec=1e-5).dataset
    assert is_primitive_cell(dataset.rotations), (
        "Primitive structure misidentified as non-primitive "
        f"(dataset type: {type(dataset).__name__})"
    )


def test_ase_atoms_to_phonopy_atoms_drops_noise_level_magnetic_moments():
    """Round-trip noise in the moments must not trigger the magnetic branch."""
    atoms = Atoms(
        "SiO",
        cell=[[4.0, 0, 0], [0, 5.0, 0], [0.6, 0, 7.0]],
        scaled_positions=[[0.0, 0.0, 0.0], [0.31, 0.5, 0.27]],
        pbc=True,
    )
    atoms.set_initial_magnetic_moments([1e-12, 1e-9])

    converted = ase_atoms_to_phonopy_atoms(atoms)

    assert converted.magnetic_moments is None
