"""
Test daughter space groups using supercell-based labels from spgrep-modulation.

Supercell label method:
1. Build modulated supercells using Modulation.get_high_symmetry_modulated_supercells()
2. Identify spacegroup using spglib on the modulated structure
3. Verify spglib confirms the structure has claimed space group (self-consistency)

These supercell-derived labels should match the output of IrRepsIrrep daughter SG
identification.

Run with: pytest tests/test_daughter_supercell_labels.py -v
"""

import pytest
import numpy as np
from pathlib import Path


def get_phonopy_example():
    """Load the chiral phonon example phonopy object."""
    from phonopy import load
    
    example_path = Path(__file__).parent.parent / "examples" / "5_chiral_phonon" / "phonopy_params.yaml"
    ph = load(str(example_path))
    return ph


def is_valid_spacegroup(sg_num):
    """Check if a space group number is valid (1-230)."""
    return 1 <= sg_num <= 230


def get_supercell_daughters(ph, qpoint, supercell_matrix):
    """
    Get daughter space group labels using modulated supercells (spgrep-modulation).
    
    This method:
    1. Uses Modulation to get eigenspaces and irreps
    2. For each eigenspace, applies OPD along standard basis directions
    3. Identifies the resulting daughter space groups
    
    Note: We use standard basis OPDs (1,0,...), (0,1,...), etc. rather than
    IsotropyEnumerator's OPDs because the latter may not give the highest
    symmetry daughters.
    
    Parameters
    ----------
    ph : Phonopy
        Phonopy object with dynamical matrix
    qpoint : array
        Q-point in fractional coordinates
    supercell_matrix : array
        Supercell matrix for modulation
        
    Returns
    -------
    list of tuples
        Each tuple: (freq, dim, set of (sg_num, sg_sym) pairs)
    """
    from spgrep_modulation.modulation import Modulation
    import spglib
    
    md = Modulation.with_supercell_and_symmetry_search(
        dynamical_matrix=ph.dynamical_matrix,
        supercell_matrix=supercell_matrix,
        qpoint=np.array(qpoint),
        factor=ph.unit_conversion_factor,
        symprec=1e-5,
    )
    
    results = []
    for i, (eigval, eigvecs, irrep) in enumerate(md.eigenspaces):
        freq = md.eigvals_to_frequencies(eigval)
        dim = eigvecs.shape[0]
        
        # For each dimension, try standard basis OPD
        sg_set = set()
        for j in range(dim):
            # Standard basis vector
            opd = np.zeros(dim, dtype=complex)
            opd[j] = 1.0
            
            # Apply modulation with this OPD
            amplitudes = list(np.abs(opd) * 0.1)
            arguments = list(np.angle(opd))
            
            try:
                cell, mod = md.get_modulated_supercell_and_modulation(
                    frequency_index=i,
                    amplitudes=amplitudes,
                    arguments=arguments,
                    return_cell=True,
                )
                
                dataset = spglib.get_symmetry_dataset(
                    (cell.cell, cell.scaled_positions, cell.numbers),
                    symprec=1e-5
                )
                if dataset is not None:
                    sg_set.add((dataset.number, dataset.international))
            except Exception:
                pass
        
        results.append((freq, dim, sg_set))
    
    return results


def validate_supercell_self_consistency(cell, expected_sg_num, symprec=1e-5):
    """
    Verify that a modulated structure has the claimed space group.
    
    This validates the supercell label by:
    1. Getting symmetry operations from the structure
    2. Re-identifying space group from those operations
    3. Confirming they match
    
    Parameters
    ----------
    cell : PhonopyAtoms
        The modulated cell
    expected_sg_num : int
        Expected space group number
    symprec : float
        Symmetry tolerance
        
    Returns
    -------
    bool
        True if self-consistent
    """
    import spglib
    
    # Get the space group from the structure
    dataset = spglib.get_symmetry_dataset(
        (cell.cell, cell.scaled_positions, cell.numbers),
        symprec=symprec
    )
    
    # Verify space group number matches
    if dataset.number != expected_sg_num:
        return False
    
    # Re-identify from symmetry operations
    sg_type = spglib.get_spacegroup_type_from_symmetry(
        dataset.rotations,
        dataset.translations,
        lattice=cell.cell,
        symprec=symprec
    )
    
    if sg_type is None:
        return False
    
    return sg_type.number == expected_sg_num


class TestSupercellLabelSelfConsistency:
    """Test that supercell label method is self-consistent."""
    
    def test_z_point_supercell_label_consistency(self):
        """Verify Z point supercell daughters are self-consistent."""
        from spgrep_modulation.modulation import Modulation
        import spglib
        
        ph = get_phonopy_example()
        qpoint = np.array([0, 0, 0.5])
        supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        
        md = Modulation.with_supercell_and_symmetry_search(
            dynamical_matrix=ph.dynamical_matrix,
            supercell_matrix=supercell_matrix,
            qpoint=qpoint,
            factor=ph.unit_conversion_factor,
            symprec=1e-5,
        )
        
        # Check first 5 eigenspaces
        for i in range(min(5, len(md.eigenspaces))):
            eigval, eigvecs, irrep = md.eigenspaces[i]
            cells = md.get_high_symmetry_modulated_supercells(i)
            
            for cell in cells:
                dataset = spglib.get_symmetry_dataset(
                    (cell.cell, cell.scaled_positions, cell.numbers),
                    symprec=1e-5
                )
                
                # Verify self-consistency
                is_consistent = validate_supercell_self_consistency(
                    cell, dataset.number
                )
                assert is_consistent, (
                    f"Eigenspace {i}: supercell label not self-consistent for "
                    f"{dataset.international} (#{dataset.number})"
                )


class TestZPointDaughterSG:
    """Test daughter space groups at Z point."""
    
    @pytest.fixture(scope="class")
    def supercell_labels_z(self):
        """Get supercell daughter SGs for Z point."""
        ph = get_phonopy_example()
        qpoint = [0, 0, 0.5]
        supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        return get_supercell_daughters(ph, qpoint, supercell_matrix)
    
    def test_z_point_has_eigenspaces(self, supercell_labels_z):
        """Verify Z point has eigenspaces."""
        assert len(supercell_labels_z) > 0, "No eigenspaces found at Z point"
    
    def test_z_point_daughters_are_valid_sgs(self, supercell_labels_z):
        """Verify all daughter space groups are valid."""
        for freq, dim, sg_set in supercell_labels_z:
            for sg_num, sg_sym in sg_set:
                assert is_valid_spacegroup(sg_num), (
                    f"Invalid space group {sg_sym} (#{sg_num}) at freq={freq:.4f} THz"
                )
    
    def test_z_point_has_chiral_daughters(self, supercell_labels_z):
        """
        Verify Z point has chiral daughters.
        
        For SG 86 (P4_2/n) at Z point, the supercell labels show:
        - P4_1 (#76) and P4_3 (#78) - chiral Sohncke Class II (enantiomorphous pair)
        - P2 (#3) and P2_1 (#4) - chiral Sohncke Class III
        
        This confirms the implementation is correct for chiral phonon transitions.
        """
        from symphon.chiral import is_sohncke
        
        has_chiral = False
        for freq, dim, sg_set in supercell_labels_z:
            for sg_num, sg_sym in sg_set:
                if is_sohncke(sg_num):
                    has_chiral = True
                    break
            if has_chiral:
                break
        
        assert has_chiral, (
            "Expected chiral daughter space groups at Z point (P4_1/P4_3, P2, P2_1). "
            "This confirms SG 86 can have chiral phonon transitions."
        )
    
    def test_z_point_expected_daughters(self, supercell_labels_z):
        """
        Verify Z point daughters are from expected set.
        
        Expected: P4_1 (#76), P4_3 (#78), P2 (#3), P2_1 (#4)
        """
        expected_sg_nums = {76, 78, 3, 4}
        
        all_found = set()
        for freq, dim, sg_set in supercell_labels_z:
            for sg_num, sg_sym in sg_set:
                all_found.add(sg_num)
        
        # All found SGs should be in expected set
        unexpected = all_found - expected_sg_nums
        assert len(unexpected) == 0, (
            f"Unexpected daughter space groups found: {unexpected}. "
            f"Expected only {expected_sg_nums}"
        )


class TestAPointDaughterSG:
    """Test daughter space groups at A point."""
    
    @pytest.fixture(scope="class")
    def supercell_labels_a(self):
        """Get supercell daughter SGs for A point."""
        ph = get_phonopy_example()
        qpoint = [0.5, 0.5, 0.5]
        supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        return get_supercell_daughters(ph, qpoint, supercell_matrix)
    
    def test_a_point_has_eigenspaces(self, supercell_labels_a):
        """Verify A point has eigenspaces."""
        assert len(supercell_labels_a) > 0, "No eigenspaces found at A point"
    
    def test_a_point_daughters_are_valid_sgs(self, supercell_labels_a):
        """Verify all daughter space groups are valid."""
        for freq, dim, sg_set in supercell_labels_a:
            for sg_num, sg_sym in sg_set:
                assert is_valid_spacegroup(sg_num), (
                    f"Invalid space group {sg_sym} (#{sg_num}) at freq={freq:.4f} THz"
                )


class TestImplementationVsSupercellLabels:
    """Test that implementation matches supercell-derived labels."""
    
    @pytest.fixture(scope="class")
    def supercell_labels_z(self):
        """Get supercell daughter SGs for Z point."""
        ph = get_phonopy_example()
        qpoint = [0, 0, 0.5]
        supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        return get_supercell_daughters(ph, qpoint, supercell_matrix)
    
    def test_implementation_daughters_match_supercell_labels_z(self, supercell_labels_z):
        """
        Test that IrRepsIrrep implementation matches supercell labels at Z point.
    
        This test will initially fail, revealing the bug in the current implementation.
        """
        from symphon.irreps.backend import IrRepsIrrep
    
        ph = get_phonopy_example()
        qpoint = [0, 0, 0.5]
    
        # Get frequencies and eigenvectors
        freqs, eigvecs = ph.get_frequencies_with_eigenvectors(qpoint)
    
        # Run implementation
        irr = IrRepsIrrep(ph.primitive, qpoint, freqs, eigvecs, log_level=0)
        irr.run()

        
        # Access the results from _irreps attribute
        # _irreps is a list of dicts with keys: "label", "opd", "opd_num", "daughter_sg"
        irreps = irr._irreps
        
        # Compare with supercell labels
        # Group implementation results by frequency
        impl_results = {}  # freq -> set of sg_nums
        band_idx = 0
        for irrep_dict in irreps:
            sg_str = irrep_dict.get("daughter_sg", "-")
            if sg_str != "-":
                # Extract SG number from string like "P4_3(#78)"
                try:
                    sg_num = int(sg_str.split("(#")[1].rstrip(")"))
                    f = round(freqs[band_idx], 4)
                    if f not in impl_results:
                        impl_results[f] = set()
                    impl_results[f].add(sg_num)
                except (IndexError, ValueError):
                    pass
            band_idx += 1
        
        # Group supercell labels by frequency
        sc_results = {}  # freq -> set of sg_nums
        for freq, dim, sg_set in supercell_labels_z:
            f = round(freq, 4)
            sc_results[f] = {sg_num for sg_num, sg_sym in sg_set}
        
        # Compare for first few frequencies
        freqs_to_check = sorted(sc_results.keys())[:5]
        
        for f in freqs_to_check:
            sc_sgs = sc_results.get(f, set())
            impl_sgs = impl_results.get(f, set())
            
            assert sc_sgs == impl_sgs, (
                f"Frequency {f:.4f} THz: implementation gives {impl_sgs}, "
                f"supercell labels give {sc_sgs}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
