"""
Tests for SAMCalculator — Phonon Spin Angular Momentum.

Covers story-003 acceptance criteria:
- AC1: Correct SAM formula S = hbar * sum Im(e* x e)
- AC2: Linear mode → SAM = 0
- AC3: Circular mode → |l| = 1 (S = ±hbar)
- AC4: Elliptical mode → 0 < |l| < 1
- Geometric circularity (mass-independent)

Run with: pytest tests/test_sam.py -v
"""

import pytest
import numpy as np

import sys
from pathlib import Path

# Direct import of sam.py to avoid the package import chain
# (circular.py -> projection.py -> irreptables.irreps, a pre-existing
# import breakage unrelated to SAMCalculator tests).
_sam_path = Path(__file__).resolve().parent.parent / "symphon" / "chiral" / "sam.py"
import importlib.util

_spec = importlib.util.spec_from_file_location("sam", _sam_path)
_sam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sam)
SAMCalculator = _sam.SAMCalculator


class TestSAMLinearMode:
    """AC2: Purely linear modes must give SAM = 0."""

    def test_linear_real_displacement(self):
        d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        sam = SAMCalculator.calculate(d, mass_weighted=True)
        assert np.allclose(sam, 0.0)

    def test_linear_single_atom(self):
        d = np.array([[1.0, 0.0, 0.0]])
        sam = SAMCalculator.calculate(d, mass_weighted=True)
        assert np.allclose(sam, 0.0)

    def test_linear_flat_input(self):
        d = np.array([1.0, 0.0, 0.0, 0.5, 0.5, 0.0])
        sam = SAMCalculator.calculate(d, mass_weighted=True)
        assert np.allclose(sam, 0.0)

    def test_linear_with_complex_but_collinear(self):
        d = np.array([[1.0 + 0.0j, 0.0, 0.0], [0.5 + 0.0j, 0.0, 0.0]])
        sam = SAMCalculator.calculate(d, mass_weighted=True)
        assert np.allclose(sam, 0.0)

    def test_linear_collinear_different_magnitudes(self):
        d = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        sam = SAMCalculator.calculate(d, mass_weighted=True)
        assert np.allclose(sam, 0.0)

    def test_linear_same_direction_at_different_atoms(self):
        d = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        sam = SAMCalculator.calculate(d, mass_weighted=True)
        assert np.allclose(sam, 0.0)


class TestSAMCircularMode:
    """AC3: Perfectly circular modes → |l| = 1, S = ±hbar.

    A circular mode for a single atom has epsilon = (1/sqrt(2), i/sqrt(2), 0),
    giving Im(e* x e) = (0, 0, 1) per atom, so S = hbar*(0,0,1), |l|=1.
    """

    def test_single_atom_circular_right(self):
        d = np.array([[1.0, 1.0j, 0.0]]) / np.sqrt(2)
        sam = SAMCalculator.calculate(d, mass_weighted=True, hbar=1.0)
        assert np.allclose(sam, [0.0, 0.0, 1.0])
        assert np.isclose(SAMCalculator.get_circularity(sam), 1.0)

    def test_single_atom_circular_left(self):
        d = np.array([[1.0, -1.0j, 0.0]]) / np.sqrt(2)
        sam = SAMCalculator.calculate(d, mass_weighted=True, hbar=1.0)
        assert np.allclose(sam, [0.0, 0.0, -1.0])
        assert np.isclose(SAMCalculator.get_circularity(sam), 1.0)

    def test_circular_xy_plane(self):
        d = np.array([[1.0, 1.0j, 0.0]]) / np.sqrt(2)
        sam = SAMCalculator.calculate(d, mass_weighted=True)
        circularity = SAMCalculator.get_circularity(sam)
        assert np.isclose(circularity, 1.0)

    def test_circular_xz_plane(self):
        d = np.array([[1.0, 0.0, 1.0j]]) / np.sqrt(2)
        sam = SAMCalculator.calculate(d, mass_weighted=True)
        circularity = SAMCalculator.get_circularity(sam)
        assert np.isclose(circularity, 1.0)
        assert sam[1] != 0  # SAM should point along y

    def test_circular_yz_plane(self):
        d = np.array([[0.0, 1.0, 1.0j]]) / np.sqrt(2)
        sam = SAMCalculator.calculate(d, mass_weighted=True)
        circularity = SAMCalculator.get_circularity(sam)
        assert np.isclose(circularity, 1.0)

    def test_two_atoms_circular(self):
        d = np.array([
            [1.0, 1.0j, 0.0],
            [1.0, 1.0j, 0.0],
        ]) / np.sqrt(2)
        sam = SAMCalculator.calculate(d, mass_weighted=True, hbar=1.0)
        assert np.allclose(sam, [0.0, 0.0, 2.0])

    def test_hbar_scaling(self):
        d = np.array([[1.0, 1.0j, 0.0]]) / np.sqrt(2)
        sam = SAMCalculator.calculate(d, mass_weighted=True, hbar=2.0)
        assert np.allclose(sam, [0.0, 0.0, 2.0])


class TestSAMEllipticalMode:
    """AC4: Elliptical modes → 0 < |l| < 1."""

    def test_elliptical_partial_phase(self):
        d = np.array([[1.0, 0.5j, 0.0]])
        d = d / np.linalg.norm(d)
        sam = SAMCalculator.calculate(d, mass_weighted=True)
        circularity = SAMCalculator.get_circularity(sam)
        assert 0.0 < circularity < 1.0
        assert circularity > 0.4

    def test_elliptical_approaching_linear(self):
        d = np.array([[1.0, 0.01j, 0.0]])
        sam = SAMCalculator.calculate(d, mass_weighted=True)
        circularity = SAMCalculator.get_circularity(sam)
        assert 0.0 < circularity < 0.1

    def test_elliptical_approaching_circular(self):
        d = np.array([[1.0, 0.99j, 0.0]])
        d = d / np.linalg.norm(d)
        sam = SAMCalculator.calculate(d, mass_weighted=True)
        circularity = SAMCalculator.get_circularity(sam)
        assert 0.9 < circularity < 1.0

    def test_elliptical_with_mixed_components(self):
        d = np.array([[1.0, 0.3 + 0.7j, 0.0]])
        d = d / np.linalg.norm(d)
        sam = SAMCalculator.calculate(d, mass_weighted=True)
        circularity = SAMCalculator.get_circularity(sam)
        assert 0.0 < circularity


class TestSAMFormula:
    """AC1: Verify the formula S = hbar * sum Im(e* x e)."""

    def test_manual_cross_product_single_atom(self):
        e = np.array([[3.0 + 4.0j, 1.0 - 2.0j, 0.5 + 0.5j]])
        e_star = e.conj()
        expected_cross = np.cross(e_star, e)
        expected_sam = np.sum(np.imag(expected_cross), axis=0)

        sam = SAMCalculator.calculate(e, mass_weighted=True, hbar=1.0)
        assert np.allclose(sam, expected_sam)

    def test_manual_cross_product_two_atoms(self):
        e = np.array([
            [1.0 + 1.0j, 0.0, 0.0],
            [0.0, 0.5 + 0.5j, 0.0],
        ])
        e_star = e.conj()
        expected_cross = np.cross(e_star, e)
        expected_sam = np.sum(np.imag(expected_cross), axis=0)

        sam = SAMCalculator.calculate(e, mass_weighted=True, hbar=1.0)
        assert np.allclose(sam, expected_sam)

    def test_hbar_applied_correctly(self):
        e = np.array([[1.0, 1.0j, 0.0]]) / np.sqrt(2)
        sam_1 = SAMCalculator.calculate(e, mass_weighted=True, hbar=1.0)
        sam_2 = SAMCalculator.calculate(e, mass_weighted=True, hbar=3.5)
        assert np.allclose(sam_2, 3.5 * sam_1)


class TestSAMPhysicalDisplacement:
    """Tests for mass_weighted=False path."""

    def test_physical_displacement_with_masses(self):
        u = np.array([[1.0, 1.0j, 0.0]]) / np.sqrt(2)
        masses = np.array([2.0])
        sam = SAMCalculator.calculate(u, masses=masses, mass_weighted=False, hbar=1.0)
        expected = masses[0] * np.array([0.0, 0.0, 1.0])
        assert np.allclose(sam, expected)

    def test_physical_displacement_normalized(self):
        u = np.array([[1.0, 1.0j, 0.0]]) / np.sqrt(2)
        masses = np.array([2.0])
        sam = SAMCalculator.calculate(
            u, masses=masses, mass_weighted=False, normalize=True, hbar=1.0
        )
        assert np.isclose(np.linalg.norm(sam), 1.0)

    def test_physical_displacement_no_masses_fallback(self):
        u = np.array([[1.0, 1.0j, 0.0]]) / np.sqrt(2)
        sam = SAMCalculator.calculate(u, mass_weighted=False, hbar=1.0)
        assert np.allclose(sam, [0.0, 0.0, 1.0])


class TestGeometricCircularity:
    """Tests for mass-independent geometric circularity."""

    def test_geometric_circular_for_circular_mode(self):
        e = np.array([[1.0, 1.0j, 0.0]]) / np.sqrt(2)
        masses = np.array([1.0])
        l_geom = SAMCalculator.calculate_geometric_circularity(e, masses)
        assert np.isclose(np.linalg.norm(l_geom), 1.0, atol=1e-10)

    def test_geometric_circular_for_linear_mode(self):
        e = np.array([[1.0, 0.0, 0.0]])
        masses = np.array([1.0])
        l_geom = SAMCalculator.calculate_geometric_circularity(e, masses)
        assert np.allclose(l_geom, 0.0)

    def test_geometric_mass_independence(self):
        e = np.array([[1.0, 1.0j, 0.0]]) / np.sqrt(2)
        masses_light = np.array([1.0])
        masses_heavy = np.array([100.0])
        l1 = SAMCalculator.calculate_geometric_circularity(e, masses_light)
        l2 = SAMCalculator.calculate_geometric_circularity(e, masses_heavy)
        assert np.allclose(l1, l2)

    def test_geometric_two_atoms_different_masses(self):
        e = np.array([
            [1.0, 1.0j, 0.0],
            [0.5, 0.5j, 0.0],
        ])
        masses = np.array([2.0, 50.0])
        l_geom = SAMCalculator.calculate_geometric_circularity(e, masses)
        assert np.linalg.norm(l_geom) > 0


class TestGetCircularity:
    """Tests for get_circularity scalar extraction."""

    def test_circularity_zero_for_zero_sam(self):
        sam = np.array([0.0, 0.0, 0.0])
        assert SAMCalculator.get_circularity(sam) == 0.0

    def test_circularity_one_for_unit_sam(self):
        sam = np.array([0.0, 0.0, 1.0])
        assert np.isclose(SAMCalculator.get_circularity(sam), 1.0)

    def test_circularity_along_axis(self):
        sam = np.array([0.3, 0.4, 1.0])
        axis = np.array([0.0, 0.0, 1.0])
        circ = SAMCalculator.get_circularity(sam, axis=axis)
        assert np.isclose(circ, 1.0)

    def test_circularity_perpendicular_axis(self):
        sam = np.array([0.0, 0.0, 1.0])
        axis = np.array([1.0, 0.0, 0.0])
        circ = SAMCalculator.get_circularity(sam, axis=axis)
        assert np.isclose(circ, 0.0)
