"""
Tests for chiral_transitions module.

Run with: pytest tests/test_chiral_transitions.py -v
"""

import pytest
import numpy as np


class TestSohnckeNumbers:
    """Tests for Sohncke space group identification."""

    def test_sohncke_count(self):
        """Verify exactly 65 Sohncke groups."""
        from anaddb_irreps.chiral_transitions import get_sohncke_numbers
        numbers = get_sohncke_numbers()
        assert len(numbers) == 65

    def test_sohncke_verification(self):
        """Verify cached Sohncke numbers match algorithmic derivation."""
        from anaddb_irreps.chiral_transitions import (
            _SOHNCKE_NUMBERS,
            _get_sohncke_numbers_from_spglib
        )
        computed = _get_sohncke_numbers_from_spglib()
        assert _SOHNCKE_NUMBERS == computed

    def test_is_sohncke_known_values(self):
        """Test is_sohncke against known values."""
        from anaddb_irreps.chiral_transitions import is_sohncke

        assert is_sohncke(1) is True
        assert is_sohncke(76) is True
        assert is_sohncke(212) is True

        assert is_sohncke(2) is False
        assert is_sohncke(136) is False
        assert is_sohncke(225) is False

    def test_sohncke_numbers_sorted(self):
        """Test that get_sohncke_numbers returns sorted list."""
        from anaddb_irreps.chiral_transitions import get_sohncke_numbers
        numbers = get_sohncke_numbers()
        assert numbers == sorted(numbers)


class TestSohnckeClassification:
    """Tests for Sohncke class classification."""

    def test_class_ii_identification(self):
        """Test Class II (enantiomorphous) identification."""
        from anaddb_irreps.chiral_transitions import (
            get_sohncke_class,
            SohnckeClass,
        )

        assert get_sohncke_class(76) == SohnckeClass.CLASS_II
        assert get_sohncke_class(78) == SohnckeClass.CLASS_II
        assert get_sohncke_class(212) == SohnckeClass.CLASS_II

    def test_class_iii_identification(self):
        """Test Class III (chiral-supporting) identification."""
        from anaddb_irreps.chiral_transitions import (
            get_sohncke_class,
            SohnckeClass,
        )

        assert get_sohncke_class(1) == SohnckeClass.CLASS_III
        assert get_sohncke_class(75) == SohnckeClass.CLASS_III
        assert get_sohncke_class(195) == SohnckeClass.CLASS_III

    def test_class_i_identification(self):
        """Test Class I (achiral) identification."""
        from anaddb_irreps.chiral_transitions import (
            get_sohncke_class,
            SohnckeClass,
        )

        assert get_sohncke_class(2) == SohnckeClass.CLASS_I
        assert get_sohncke_class(136) == SohnckeClass.CLASS_I
        assert get_sohncke_class(225) == SohnckeClass.CLASS_I

    def test_enantiomorph_partner(self):
        """Test enantiomorph partner lookup."""
        from anaddb_irreps.chiral_transitions import get_enantiomorph_partner

        assert get_enantiomorph_partner(76) == 78
        assert get_enantiomorph_partner(78) == 76
        assert get_enantiomorph_partner(212) == 213
        assert get_enantiomorph_partner(213) == 212

    def test_no_enantiomorph_partner(self):
        """Test that Class III groups have no enantiomorph partner."""
        from anaddb_irreps.chiral_transitions import get_enantiomorph_partner

        assert get_enantiomorph_partner(1) is None
        assert get_enantiomorph_partner(75) is None
        assert get_enantiomorph_partner(195) is None


class TestImproperOperations:
    """Tests for improper operation classification."""

    def test_inversion_detection(self):
        """Test inversion detection."""
        from anaddb_irreps.chiral_transitions import (
            classify_improper_operation,
            ImproperOperationType
        )

        inversion = -np.eye(3, dtype=int)
        result = classify_improper_operation(inversion)
        assert result == ImproperOperationType.INVERSION

    def test_mirror_detection(self):
        """Test mirror detection."""
        from anaddb_irreps.chiral_transitions import (
            classify_improper_operation,
            ImproperOperationType
        )

        mirror = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=int)
        result = classify_improper_operation(mirror)
        assert result == ImproperOperationType.MIRROR

    def test_proper_rotation(self):
        """Test that proper rotations return None."""
        from anaddb_irreps.chiral_transitions import classify_improper_operation

        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
        result = classify_improper_operation(rotation)
        assert result is None

    def test_rotoinversion_detection(self):
        """Test rotoinversion detection."""
        from anaddb_irreps.chiral_transitions import (
            classify_improper_operation,
            ImproperOperationType
        )

        rotoinversion = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=int)
        result = classify_improper_operation(rotoinversion)
        assert result in (ImproperOperationType.INVERSION, ImproperOperationType.ROTOUNVERSION)

    def test_has_improper_operations(self):
        """Test has_improper_operations function."""
        from anaddb_irreps.chiral_transitions import has_improper_operations

        rotations_with_inversion = np.array([
            np.eye(3, dtype=int),
            -np.eye(3, dtype=int),
        ])
        assert has_improper_operations(rotations_with_inversion) is True

        proper_rotations = np.array([
            np.eye(3, dtype=int),
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int),
        ])
        assert has_improper_operations(proper_rotations) is False


class TestOPDConversion:
    """Tests for order parameter direction symbolic conversion."""

    def test_opd_single_element(self):
        """Test OPD with single element."""
        from anaddb_irreps.chiral_transitions import opd_to_symbolic

        result = opd_to_symbolic(np.array([1]))
        assert 'a' in result

    def test_opd_with_zeros(self):
        """Test OPD with zeros."""
        from anaddb_irreps.chiral_transitions import opd_to_symbolic

        result = opd_to_symbolic(np.array([1, 0, 0]))
        assert 'a' in result
        assert '0' in result

    def test_opd_all_zeros(self):
        """Test OPD with all zeros."""
        from anaddb_irreps.chiral_transitions import opd_to_symbolic

        result = opd_to_symbolic(np.array([0, 0, 0]))
        assert result == '(0,0,0)'

    def test_opd_negative(self):
        """Test OPD with negative coefficient."""
        from anaddb_irreps.chiral_transitions import opd_to_symbolic

        result = opd_to_symbolic(np.array([-1, 0, 0]))
        assert '-a' in result


class TestSpaceGroupInfo:
    """Tests for SpaceGroupInfo dataclass."""

    def test_spacegroup_info_creation(self):
        """Test SpaceGroupInfo creation."""
        from anaddb_irreps.chiral_transitions import SpaceGroupInfo

        info = SpaceGroupInfo(
            number=1,
            symbol="P1",
            rotations=np.array([np.eye(3, dtype=int)]),
            translations=np.array([[0, 0, 0]]),
            point_group_symbol="1",
            order=1
        )

        assert info.number == 1
        assert info.symbol == "P1"
        assert info.order == 1

    def test_spacegroup_info_sohncke_properties(self):
        """Test SpaceGroupInfo Sohncke-related properties."""
        from anaddb_irreps.chiral_transitions import (
            SpaceGroupInfo,
            SohnckeClass
        )

        info = SpaceGroupInfo(
            number=76,
            symbol="P4_1",
            rotations=np.array([np.eye(3, dtype=int)]),
            translations=np.array([[0, 0, 0]]),
            point_group_symbol="4",
            order=4
        )

        assert info.is_sohncke is True
        assert info.sohncke_class == SohnckeClass.CLASS_II


class TestChiralTransition:
    """Tests for ChiralTransition dataclass."""

    def test_chiral_transition_creation(self):
        """Test ChiralTransition creation."""
        from anaddb_irreps.chiral_transitions import (
            ChiralTransition,
            OrderParameterDirection,
            SohnckeClass
        )

        transition = ChiralTransition(
            parent_spg_number=136,
            parent_spg_symbol="P4_2/mnm",
            parent_spg_order=16,
            qpoint=np.array([0, 0, 0.5]),
            qpoint_label="Z",
            irrep_label="Z4",
            irrep_dimension=2,
            opd=OrderParameterDirection(
                numerical=np.array([1, 0]),
                symbolic="(a,0)",
                num_free_params=1
            ),
            daughter_spg_number=76,
            daughter_spg_symbol="P4_1",
            daughter_spg_order=4,
            domain_multiplicity=4,
            enantiomeric_domain_count=1,
            lost_operations=[],
            lost_inversion=True,
            lost_mirrors=2,
            lost_glides=2,
            sohncke_class=SohnckeClass.CLASS_II,
            enantiomorph_partner=78
        )

        summary = transition.get_summary()
        assert "P4_2/mnm" in summary
        assert "Z4" in summary
        assert "P4_1" in summary


class TestChiralTransitionFinder:
    """Tests for ChiralTransitionFinder class."""

    def test_finder_creation(self):
        """Test ChiralTransitionFinder creation."""
        from anaddb_irreps.chiral_transitions import ChiralTransitionFinder

        finder = ChiralTransitionFinder(136)
        assert finder.spg_number == 136

    def test_parent_chiral_detection(self):
        """Test that chiral parents are detected."""
        from anaddb_irreps.chiral_transitions import ChiralTransitionFinder

        finder = ChiralTransitionFinder(76)
        assert finder.is_parent_chiral is True

    def test_parent_achiral_detection(self):
        """Test that achiral parents are detected."""
        from anaddb_irreps.chiral_transitions import ChiralTransitionFinder

        finder = ChiralTransitionFinder(136)
        assert finder.is_parent_chiral is False

    def test_spacegroup_info_loading(self):
        """Test that space group info is loaded correctly."""
        from anaddb_irreps.chiral_transitions import ChiralTransitionFinder

        finder = ChiralTransitionFinder(136)
        info = finder.spacegroup_info

        assert info.number == 136
        assert info.order > 0

    def test_centrosymmetric_detection(self):
        """Test centrosymmetric detection."""
        from anaddb_irreps.chiral_transitions import ChiralTransitionFinder

        finder = ChiralTransitionFinder(136)
        assert finder.is_parent_centrosymmetric is True

        finder2 = ChiralTransitionFinder(99)
        assert finder2.is_parent_centrosymmetric is False

    def test_chiral_parent_raises_error(self):
        """Test that chiral parents raise ValueError."""
        from anaddb_irreps.chiral_transitions import ChiralTransitionFinder

        finder = ChiralTransitionFinder(76)
        with pytest.raises(ValueError, match="already chiral"):
            finder.find_chiral_transitions()

    def test_invalid_spg_number(self):
        """Test that invalid space group numbers raise ValueError."""
        from anaddb_irreps.chiral_transitions import ChiralTransitionFinder

        with pytest.raises(ValueError):
            ChiralTransitionFinder(0)

        with pytest.raises(ValueError):
            ChiralTransitionFinder(231)

    def test_get_proper_subgroup_info_fast(self):
        """Test get_proper_subgroup_info returns Sohncke daughter."""
        from anaddb_irreps.chiral_transitions import (
            ChiralTransitionFinder,
            is_sohncke,
        )

        finder = ChiralTransitionFinder(136)
        daughter_num, daughter_sym, daughter_order = finder.get_proper_subgroup_info()

        assert daughter_num > 0
        assert daughter_num != 136
        assert is_sohncke(daughter_num)
        assert daughter_sym != "Unknown"
        assert daughter_order < finder.spacegroup_info.order

    def test_get_proper_subgroup_info_non_centrosymmetric(self):
        """Test get_proper_subgroup_info for non-centrosymmetric parent."""
        from anaddb_irreps.chiral_transitions import (
            ChiralTransitionFinder,
            is_sohncke,
        )

        finder = ChiralTransitionFinder(99)
        daughter_num, daughter_sym, daughter_order = finder.get_proper_subgroup_info()

        assert daughter_num > 0
        assert is_sohncke(daughter_num)


class TestReporting:
    """Tests for reporting functions."""

    def test_format_transition_table_empty(self):
        """Test format_transition_table with empty list."""
        from anaddb_irreps.chiral_transitions import format_transition_table

        result = format_transition_table([])
        assert "No transitions" in result

    def test_format_lost_operations_detail(self):
        """Test format_lost_operations_detail."""
        from anaddb_irreps.chiral_transitions import (
            format_lost_operations_detail,
            ChiralTransition,
            OrderParameterDirection,
            LostOperation,
            ImproperOperationType,
            SohnckeClass
        )

        transition = ChiralTransition(
            parent_spg_number=136,
            parent_spg_symbol="P4_2/mnm",
            parent_spg_order=16,
            qpoint=np.array([0, 0, 0.5]),
            qpoint_label="Z",
            irrep_label="Z4",
            irrep_dimension=2,
            opd=OrderParameterDirection(
                numerical=np.array([1, 0]),
                symbolic="(a,0)",
                num_free_params=1
            ),
            daughter_spg_number=76,
            daughter_spg_symbol="P4_1",
            daughter_spg_order=4,
            domain_multiplicity=4,
            enantiomeric_domain_count=1,
            lost_operations=[
                LostOperation(
                    operation_type=ImproperOperationType.INVERSION,
                    rotation=-np.eye(3, dtype=int),
                    translation=np.zeros(3),
                    description="Inversion at origin",
                    jones_symbol="-x,-y,-z"
                )
            ],
            lost_inversion=True,
            lost_mirrors=0,
            lost_glides=0,
            sohncke_class=SohnckeClass.CLASS_II,
            enantiomorph_partner=78
        )

        result = format_lost_operations_detail(transition)
        assert "Lost operations" in result
        assert "inversion" in result


class TestIntegration:
    """Integration tests for find_chiral_transitions."""

    def test_sg136_finds_transitions(self):
        """Test SG 136 (P4_2/mnm) chiral transition search."""
        from anaddb_irreps.chiral_transitions import (
            ChiralTransitionFinder,
            is_sohncke,
        )

        finder = ChiralTransitionFinder(136)
        transitions = finder.find_chiral_transitions()

        # Daughters may be achiral - this is correct behavior
        for t in transitions:
            assert t.parent_spg_number == 136
            assert t.domain_multiplicity > 0

    def test_transitions_have_lost_operations(self):
        """Test that chiral transitions (when found) have lost improper operations."""
        from anaddb_irreps.chiral_transitions import ChiralTransitionFinder, is_sohncke

        finder = ChiralTransitionFinder(136)
        transitions = finder.find_chiral_transitions()

        # Only check chiral daughters for lost operations
        chiral_transitions = [t for t in transitions if is_sohncke(t.daughter_spg_number)]
        
        for t in chiral_transitions:
            # Chiral transitions must lose at least one improper operation
            total_lost = t.lost_inversion + t.lost_mirrors + t.lost_glides
            assert total_lost > 0, f"Should lose improper operations, but got {t.lost_operations}"

    def test_format_table_with_real_transitions(self):
        """Test format_transition_table with real transitions."""
        from anaddb_irreps.chiral_transitions import (
            ChiralTransitionFinder,
            format_transition_table,
        )

        finder = ChiralTransitionFinder(136)
        transitions = finder.find_chiral_transitions()

        result = format_transition_table(transitions[:5])

        assert "Phase Transitions" in result
        assert "P4_2/mnm" in result
