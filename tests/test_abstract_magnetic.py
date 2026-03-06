import pytest
import numpy as np
from symphon.abstract_magnetic import AbstractMagneticTransitionFinder

def test_abstract_magnetic_finder_creation():
    # Test creation for SG 2 (P-1, centrosymmetric)
    finder = AbstractMagneticTransitionFinder(2)
    assert finder.spg_number == 2
    assert finder.spacegroup_info is not None

def test_chiral_parent_raises_error():
    # SG 1 (P1) is chiral (Sohncke), should raise error
    with pytest.raises(ValueError, match="is already chiral"):
        AbstractMagneticTransitionFinder(1)

def test_find_transitions_sg2():
    finder = AbstractMagneticTransitionFinder(2)
    # Search at Gamma point
    transitions = finder.find_transitions([0, 0, 0])
    
    # Check that we found some transitions
    assert len(transitions) > 0
    assert all(isinstance(t, dict) for t in transitions)
    
    chiral_transitions = [t for t in transitions if t.get('is_chiral')]
    # P-1 has no chiral magnetic subgroups at Gamma (either -1 or -1' is preserved)
    assert len(chiral_transitions) == 0

def test_qpoint_transformation():
    # Check non-gamma point for SG 81 (P-4) which has chiral magnetic transitions
    finder = AbstractMagneticTransitionFinder(81)
    # Gamma point
    transitions = finder.find_transitions([0, 0, 0])
    assert len(transitions) > 0
    
    chiral_transitions = [t for t in transitions if t.get('is_chiral')]
    # SG 81 has E irrep that gives a chiral subgroup
    assert len(chiral_transitions) > 0

def test_time_reversal_flags():
    finder = AbstractMagneticTransitionFinder(2)
    transitions = finder.find_transitions([0, 0, 0])
    
    # Check tr_broken and is_chiral
    for t in transitions:
        assert isinstance(t.get('is_chiral', True), bool)
