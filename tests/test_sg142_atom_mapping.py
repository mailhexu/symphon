"""
Test atom mapping for non-Gamma k-points in SG 142.

This tests the fix for the bug where atom mapping failed in BCS frame.
The solution: do atom mapping in primitive frame, only use BCS frame
for character table matching.

Run:
    pytest tests/test_sg142_atom_mapping.py -v
"""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
TEST_DATA = PROJECT_ROOT / "tests" / "data" / "phonopy_params.yaml.xz"


def test_sg142_n_point_no_crash():
    """Test that SG 142 N-point processing doesn't crash.
    
    This was the original bug: atom mapping failed for N-point in BCS frame.
    """
    if not TEST_DATA.exists():
        pytest.skip(f"Test data not found: {TEST_DATA}")
    
    result = subprocess.run(
        [sys.executable, "-c",
         f"from symphon.cli import main_phonopy; "
         f"import sys; sys.argv = ['phonopy-irreps', '-p', '{TEST_DATA}']; "
         f"main_phonopy()"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=120,
    )
    
    assert "Atom mapping failed" not in result.stderr, f"Atom mapping failed: {result.stderr}"
    assert "N point" in result.stdout, "N-point not found in output"


def test_sg142_n_point_bcs_labels():
    """Test that SG 142 N-point has correct BCS labels (N1)."""
    if not TEST_DATA.exists():
        pytest.skip(f"Test data not found: {TEST_DATA}")
    
    result = subprocess.run(
        [sys.executable, "-c",
         f"from symphon.cli import main_phonopy; "
         f"import sys; sys.argv = ['phonopy-irreps', '-p', '{TEST_DATA}']; "
         f"main_phonopy()"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=120,
    )
    
    output = result.stdout + result.stderr
    
    assert "N1" in output, "N1 label not found in N-point output"


def test_sg142_all_high_sym_points():
    """Test that all 5 high-symmetry points are processed for SG 142."""
    if not TEST_DATA.exists():
        pytest.skip(f"Test data not found: {TEST_DATA}")
    
    result = subprocess.run(
        [sys.executable, "-c",
         f"from symphon.cli import main_phonopy; "
         f"import sys; sys.argv = ['phonopy-irreps', '-p', '{TEST_DATA}']; "
         f"main_phonopy()"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=120,
    )
    
    output = result.stdout + result.stderr
    
    expected_points = ["GM point", "M point", "N point", "P point", "X point"]
    for point in expected_points:
        assert point in output, f"{point} not found in output"


def test_sg142_atom_mapping_primitive_frame():
    """Test atom mapping directly in primitive frame for N-point symmetry.
    
    This verifies the core fix: atom mapping works in primitive frame.
    """
    from phonopy import load as phonopy_load
    from irrep.spacegroup_irreps import SpaceGroupIrreps
    import numpy as np
    
    if not TEST_DATA.exists():
        pytest.skip(f"Test data not found: {TEST_DATA}")
    
    phonon = phonopy_load(TEST_DATA)
    positions = phonon.primitive.scaled_positions
    numbers = phonon.primitive.numbers
    cell = phonon.primitive.cell
    
    sg = SpaceGroupIrreps.from_cell(
        cell=(cell, positions, numbers),
        spinor=False,
        include_TR=False,
        search_cell=True,
        symprec=1e-5,
        verbosity=0
    )
    
    refUCinv = np.linalg.inv(sg.refUC)
    k_bcs = np.array([0.5, 0, 0.5])
    k_prim = refUCinv.T @ k_bcs
    
    little_group_indices = []
    for i_sym, sym in enumerate(sg.symmetries):
        diff = (sym.rotation @ k_prim - k_prim)
        if np.allclose(diff - np.round(diff), 0, atol=1e-5):
            little_group_indices.append(i_sym)
    
    assert len(little_group_indices) == 4, f"Expected 4 little group ops, got {len(little_group_indices)}"
    
    num_atoms = len(positions)
    symprec = 1e-5
    
    for isym in little_group_indices:
        sym = sg.symmetries[isym]
        rot = sym.rotation
        trans = sym.translation
        rot_inv = np.linalg.inv(rot)
        
        for k in range(num_atoms):
            new_pos = rot @ positions[k] + trans
            found = False
            for j in range(num_atoms):
                diff = new_pos - positions[j]
                diff_round = np.round(diff)
                if np.allclose(diff - diff_round, 0, atol=symprec):
                    found = True
                    break
            assert found, f"Atom mapping failed for sym {isym+1}, atom {k}"
