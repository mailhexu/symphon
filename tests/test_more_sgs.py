import pytest
from anaddb_irreps.chiral_transitions import ChiralTransitionFinder, SohnckeClass

# Specifically requested by the user to be verified before moving to full representation
TARGET_SGS = [
    55, 60, 84, 86, 105, 106, 131, 132, 133, 134, 135, 136, 137, 138
]

# Expected Class II (Enantiomorphous pairs) transitions
EXPECTED_CLASS_II = {
  55: [],
  60: [],
  84: [
    {"qpoint_label": "Z", "irrep_label": "Z2", "opd_symbolic": "(a,0)", "daughter_spg_number": 78},
    {"qpoint_label": "Z", "irrep_label": "Z2", "opd_symbolic": "(0,a)", "daughter_spg_number": 76}
  ],
  86: [
    {"qpoint_label": "Z", "irrep_label": "Z2", "opd_symbolic": "(a,0)", "daughter_spg_number": 78},
    {"qpoint_label": "Z", "irrep_label": "Z2", "opd_symbolic": "(0,a)", "daughter_spg_number": 76}
  ],
  105: [
    {"qpoint_label": "Z", "irrep_label": "Z5", "opd_symbolic": "(a,0)", "daughter_spg_number": 78},
    {"qpoint_label": "Z", "irrep_label": "Z5", "opd_symbolic": "(0,a)", "daughter_spg_number": 76}
  ],
  106: [
    {"qpoint_label": "Z", "irrep_label": "Z5", "opd_symbolic": "(a,0)", "daughter_spg_number": 78},
    {"qpoint_label": "Z", "irrep_label": "Z5", "opd_symbolic": "(0,a)", "daughter_spg_number": 76}
  ],
  131: [
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(a,0)", "daughter_spg_number": 95},
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(0,a)", "daughter_spg_number": 91},
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(0,-a)", "daughter_spg_number": 91},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(a,0)", "daughter_spg_number": 95},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(0,a)", "daughter_spg_number": 91}
  ],
  132: [
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(a,0)", "daughter_spg_number": 95},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(0,a)", "daughter_spg_number": 91},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(0,-a)", "daughter_spg_number": 91},
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(a,0)", "daughter_spg_number": 95},
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(0,a)", "daughter_spg_number": 91}
  ],
  133: [
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(a,0)", "daughter_spg_number": 95},
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(0,a)", "daughter_spg_number": 91},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(a,0)", "daughter_spg_number": 95},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(0,a)", "daughter_spg_number": 91},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(0,-a)", "daughter_spg_number": 91}
  ],
  134: [
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(a,0)", "daughter_spg_number": 95},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(0,a)", "daughter_spg_number": 91},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(0,-a)", "daughter_spg_number": 91},
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(a,0)", "daughter_spg_number": 95},
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(0,a)", "daughter_spg_number": 91}
  ],
  135: [
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(a,0)", "daughter_spg_number": 96},
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(0,a)", "daughter_spg_number": 92},
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(0,-a)", "daughter_spg_number": 92},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(a,0)", "daughter_spg_number": 96},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(0,a)", "daughter_spg_number": 92}
  ],
  136: [
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(a,0)", "daughter_spg_number": 96},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(0,a)", "daughter_spg_number": 92},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(0,-a)", "daughter_spg_number": 92},
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(a,0)", "daughter_spg_number": 96},
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(0,a)", "daughter_spg_number": 92}
  ],
  137: [
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(a,0)", "daughter_spg_number": 96},
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(0,a)", "daughter_spg_number": 92},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(a,0)", "daughter_spg_number": 96},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(0,a)", "daughter_spg_number": 92},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(0,-a)", "daughter_spg_number": 92}
  ],
  138: [
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(a,0)", "daughter_spg_number": 96},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(0,a)", "daughter_spg_number": 92},
    {"qpoint_label": "Z", "irrep_label": "Z3", "opd_symbolic": "(0,-a)", "daughter_spg_number": 92},
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(a,0)", "daughter_spg_number": 96},
    {"qpoint_label": "Z", "irrep_label": "Z4", "opd_symbolic": "(0,a)", "daughter_spg_number": 92}
  ]
}

@pytest.mark.parametrize("spg_num", TARGET_SGS)
def test_specific_sgs_no_crash(spg_num):
    """
    Test that the current implementation of ChiralTransitionFinder
    works without crashing for these specific space groups.
    We just verify it returns a list of transitions and matches
    the expected Class II chiral transitions.
    """
    finder = ChiralTransitionFinder(spg_num)
    transitions = finder.find_chiral_transitions()
    
    assert isinstance(transitions, list)
    
    # Filter class II transitions
    class2_trans = [t for t in transitions if t.sohncke_class == SohnckeClass.CLASS_II]
    
    actual_summaries = []
    for t in class2_trans:
        actual_summaries.append({
            "qpoint_label": t.qpoint_label,
            "irrep_label": t.irrep_label,
            "opd_symbolic": t.opd.symbolic,
            "daughter_spg_number": t.daughter_spg_number
        })
        
    expected_summaries = EXPECTED_CLASS_II[spg_num]
    
    # We assert that every expected summary is in actual_summaries.
    # We do it order-independent to avoid brittleness.
    for exp in expected_summaries:
        assert exp in actual_summaries, f"Missing expected transition {exp} for SG {spg_num}"
    
    # We also assert no unexpected class II transitions are found.
    for act in actual_summaries:
        assert act in expected_summaries, f"Unexpected transition {act} for SG {spg_num}"
