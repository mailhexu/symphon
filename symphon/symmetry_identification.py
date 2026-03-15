import numpy as np
import spglib
from typing import Optional, Tuple
from itertools import product

# Mapping from primitive to centered space group numbers for tetragonal
# When the supercell has centering, we need to find the corresponding centered SG
_CENTERING_MAP = {}  # Populated lazily

def _get_centered_sg_from_primitive(sg_num: int, centering_type: str) -> Tuple[int, str]:
    """
    Given a primitive space group number and centering type ('A', 'B', 'C', 'I', 'F'),
    find the corresponding centered space group with the same point group.
    """
    if centering_type == 'P':
        return sg_num, None
    
    try:
        sg_type = spglib.get_spacegroup_type(sg_num)
        if sg_type is None:
            return sg_num, None
        pg = sg_type.pointgroup_international if hasattr(sg_type, 'pointgroup_international') else sg_type.get('pointgroup_international', '')
        original_sym = sg_type.international_short if hasattr(sg_type, 'international_short') else sg_type.get('international_short', '')
    except Exception:
        return sg_num, None
    
    # If already the right centering, return as-is
    if original_sym.startswith(centering_type):
        return sg_num, original_sym
    
    # Search for a space group with the same point group and the right centering
    for num in range(1, 231):
        try:
            test_type = spglib.get_spacegroup_type(num)
            if test_type is None:
                continue
            test_pg = test_type.pointgroup_international if hasattr(test_type, 'pointgroup_international') else test_type.get('pointgroup_international', '')
            test_sym = test_type.international_short if hasattr(test_type, 'international_short') else test_type.get('international_short', '')
            if test_pg == pg and test_sym.startswith(centering_type):
                return num, test_sym
        except Exception:
            continue
    
    return sg_num, None


def identify_spacegroup_from_operations(
    lattice: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    symprec: float = 1e-5,
    use_supercell_type: bool = False,
    check_centering: bool = False,
    parent_lattice: Optional[np.ndarray] = None
) -> Tuple[int, str]:
    """
    Identify the space group corresponding to a set of symmetry operations.
    
    Parameters:
        use_supercell_type: If True, use get_spacegroup_type_from_symmetry to
                           identify the SG without reducing to primitive cell.
        check_centering: If True and use_supercell_type is True, check if the
                        operations include lattice centering and adjust SG accordingly.
        parent_lattice: Parent lattice for centering detection.
    """
    if use_supercell_type:
        sg_type = spglib.get_spacegroup_type_from_symmetry(
            rotations, translations, lattice=lattice, symprec=symprec
        )
        if sg_type is None:
            return 0, "Unknown"
        
        num = sg_type.number if hasattr(sg_type, 'number') else sg_type.get('number', 0)
        sym = sg_type.international_short if hasattr(sg_type, 'international_short') else sg_type.get('international_short', '')
        
        if check_centering and parent_lattice is not None and not sym.startswith('P'):
            return num, sym
        
        if check_centering and parent_lattice is not None and sym.startswith('P'):
            vol_ratio = np.linalg.det(lattice) / np.linalg.det(parent_lattice)
            if vol_ratio > 1.5:
                centering_trans = _detect_centering_translations(translations, symprec)
                if centering_trans:
                    centered_num, centered_sym = _get_centered_sg_from_primitive(num, centering_trans)
                    if centered_sym is not None:
                        return centered_num, centered_sym
        
        return num, sym
    
    seeds = [
        [0.11, 0.23, 0.37],
        [0.43, 0.51, 0.67],
        [0.79, 0.17, 0.29]
    ]
    
    all_pos = []
    numbers = []
    
    for seed_idx, seed in enumerate(seeds):
        for r, t in zip(rotations, translations):
            pos = (np.dot(r, seed) + t) % 1.0
            
            is_new = True
            for existing_pos in all_pos:
                diff = pos - existing_pos
                diff -= np.round(diff)
                if np.all(np.abs(diff) < symprec):
                    is_new = False
                    break
            
            if is_new:
                all_pos.append(pos)
                numbers.append(seed_idx + 1)
    
    if not all_pos:
        return 1, "P1"
        
    all_pos_arr = np.array(all_pos)
    numbers_arr = np.array(numbers, dtype='intc')
    cell = (lattice, all_pos_arr, numbers_arr)
    
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if dataset:
        return dataset.number, dataset.international
    return 0, "Unknown"


def _detect_centering_translations(translations: np.ndarray, symprec: float = 1e-5) -> Optional[str]:
    """Detect if translations include lattice centering (A, B, C, I, F)."""
    has_identity = False
    has_A = False
    has_B = False
    has_C = False
    has_I = False
    
    for t in translations:
        t_mod = t % 1.0
        t_mod = np.where(np.abs(t_mod - 1.0) < symprec, 0.0, t_mod)
        t_mod = np.where(np.abs(t_mod) < symprec, 0.0, t_mod)
        
        if np.allclose(t_mod, [0, 0, 0], atol=symprec):
            has_identity = True
        elif np.allclose(t_mod, [0, 0.5, 0.5], atol=symprec):
            has_A = True
        elif np.allclose(t_mod, [0.5, 0, 0.5], atol=symprec):
            has_B = True
        elif np.allclose(t_mod, [0.5, 0.5, 0], atol=symprec):
            has_C = True
        elif np.allclose(t_mod, [0.5, 0.5, 0.5], atol=symprec):
            has_I = True
    
    if not has_identity:
        return None
    
    if has_I:
        return 'I'
    if has_A:
        return 'A'
    if has_B:
        return 'B'
    if has_C:
        return 'C'
    
    return None

def _get_supercell_matrix_from_qpoint(qpoint: np.ndarray, max_denom: int = 12) -> np.ndarray:
    """Get supercell matrix S such that S @ q is commensurate (integer)."""
    denoms = [1, 1, 1]
    for i, x in enumerate(qpoint):
        if np.isclose(x, 0, atol=1e-5):
            continue
        for d in range(1, max_denom + 1):
            if np.isclose((x * d) % 1.0, 0, atol=1e-5):
                denoms[i] = d
                break
    return np.diag(denoms)


def get_isotropy_subgroup(
    parent_lattice: np.ndarray,
    parent_rotations: np.ndarray,
    parent_translations: np.ndarray,
    qpoint: np.ndarray,
    irrep_matrices: np.ndarray,
    opd: np.ndarray,
    symprec: float = 1e-5,
    return_indices: bool = False
) -> Tuple:
    """
    Find the isotropy subgroup for a given OPD of an irrep at q.
    
    For zone-boundary q (q != 0), expands to a commensurate supercell
    to correctly identify the daughter space group with proper lattice centering.
    
    Key insight for zone-boundary modes: operations that don't preserve the OPD
    in the primitive cell CAN preserve it when combined with appropriate lattice
    translations in the supercell. For example, inversion combined with a half-cell
    translation can preserve an antiphase distortion pattern.
    """
    if np.allclose(qpoint, 0, atol=1e-5):
        preserved_indices = []
        for i, mat in enumerate(irrep_matrices):
            phase = np.exp(-2j * np.pi * np.dot(qpoint, parent_translations[i]))
            expected = opd * phase
            actual = np.dot(mat, opd)
            if np.allclose(actual, expected, atol=1e-4):
                preserved_indices.append(i)
        
        if not preserved_indices:
            if return_indices:
                return 1, "P1", []
            return 1, "P1"
        
        sub_rots = parent_rotations[preserved_indices]
        sub_trans = parent_translations[preserved_indices]
        sg_num, sg_sym = identify_spacegroup_from_operations(parent_lattice, sub_rots, sub_trans, symprec)
        if return_indices:
            return sg_num, sg_sym, preserved_indices
        return sg_num, sg_sym
    
    S = _get_supercell_matrix_from_qpoint(qpoint)
    S_inv = np.linalg.inv(S)
    sc_lattice = np.dot(S, parent_lattice)
    
    M = int(np.max(np.diag(S))) + 2
    lattice_trans_n = []
    for nx, ny, nz in product(range(-M, M+1), repeat=3):
        n = np.array([nx, ny, nz])
        n_sc = np.dot(S_inv, n)
        if np.all(n_sc > -1e-5) and np.all(n_sc < 1 - 1e-5):
            lattice_trans_n.append(n)
    
    sc_rots = []
    sc_trans = []
    preserved_indices = []
    
    for j, (r, t, mat_j) in enumerate(zip(parent_rotations, parent_translations, irrep_matrices)):
        found_preservation = False
        
        for n in lattice_trans_n:
            phase = np.exp(-2j * np.pi * np.dot(qpoint, n))
            mat = mat_j * phase
            diff = np.linalg.norm(np.dot(mat, opd) - opd)
            
            if diff < 1e-4:
                r_prime = np.dot(S_inv, np.dot(r, S))
                t_prime = np.dot(S_inv, t + n)
                sc_rots.append(r_prime)
                sc_trans.append(t_prime)
                found_preservation = True
        
        if found_preservation:
            preserved_indices.append(j)
    
    if not sc_rots:
        if return_indices:
            return 1, "P1", preserved_indices
        return 1, "P1"
    
    sc_rots_arr = np.round(sc_rots).astype('intc')
    sc_trans_arr = np.array(sc_trans, dtype='double')
    
    sg_num, sg_sym = identify_spacegroup_from_operations(
        sc_lattice, sc_rots_arr, sc_trans_arr, symprec,
        use_supercell_type=True,
        check_centering=True,
        parent_lattice=parent_lattice
    )
    if return_indices:
        return sg_num, sg_sym, preserved_indices
    return sg_num, sg_sym


def get_isotropy_subgroup_from_supercell(
    cell: Tuple[np.ndarray, np.ndarray, np.ndarray],
    qpoint: np.ndarray,
    opd: np.ndarray,
    irrep_matrices: np.ndarray,
    symprec: float = 1e-5,
) -> Tuple[int, str]:
    """
    Alternative method: Generate a real supercell with the distortion and use spglib
    to identify the daughter space group.
    
    This is useful for verifying the symmetry-based calculation.
    
    Parameters:
        cell: Tuple of (lattice, positions, numbers) for the primitive cell
        qpoint: The q-point (fractional coordinates)
        opd: The order parameter direction (normalized eigenvector component)
        irrep_matrices: The irrep matrices for the little group
        symprec: Symmetry precision for spglib
        
    Returns:
        (sg_number, sg_symbol) of the daughter space group
    """
    from spgrep import get_spacegroup_irreps_from_primitive_symmetry
    from spgrep.group import get_little_group
    
    lattice, positions, numbers = cell
    
    # Get supercell matrix
    S = _get_supercell_matrix_from_qpoint(qpoint)
    S_inv = np.linalg.inv(S)
    sc_lattice = np.dot(S, lattice)
    
    # Determine supercell size
    sc_size = int(np.round(np.linalg.det(S)))
    
    # Build the supercell
    sc_positions = []
    sc_numbers = []
    
    # Generate all lattice translations in the supercell
    lattice_points = []
    for nx in range(int(S[0, 0])):
        for ny in range(int(S[1, 1])):
            for nz in range(int(S[2, 2])):
                n = np.array([nx, ny, nz], dtype=float)
                # Transform to supercell fractional coordinates
                n_sc = np.dot(S_inv, n)
                lattice_points.append((n, n_sc))
    
    # Place atoms in the supercell
    for atom_idx, (pos, num) in enumerate(zip(positions, numbers)):
        for n, n_sc in lattice_points:
            # Position in supercell fractional coordinates
            sc_pos = np.dot(S_inv, pos + n)
            sc_positions.append(sc_pos % 1.0)
            sc_numbers.append(num)
    
    sc_positions = np.array(sc_positions)
    sc_numbers = np.array(sc_numbers, dtype='intc')
    
    # Get parent symmetry operations
    parent_dataset = spglib.get_symmetry_dataset((lattice, positions, numbers), symprec=symprec)
    if parent_dataset is None:
        return 0, "Unknown"
    
    parent_rots = parent_dataset.rotations
    parent_trans = parent_dataset.translations
    
    # Get little group
    _, _, lg_indices = get_little_group(parent_rots, parent_trans, qpoint, symprec)
    
    if len(lg_indices) == 0:
        return 1, "P1"
    
    # Get preserved operations based on isotropy condition
    lg_rots = parent_rots[lg_indices]
    lg_trans = parent_trans[lg_indices]
    
    # For each operation, check if it preserves the OPD
    preserved_rots = []
    preserved_trans = []
    
    for i, (rot, trans) in enumerate(zip(lg_rots, lg_trans)):
        # Calculate phase factor
        phase = np.exp(-2j * np.pi * np.dot(qpoint, trans))
        
        # Get irrep matrix for this operation
        mat = irrep_matrices[i]
        
        # Check isotropy condition: mat @ opd ≈ opd * phase
        expected = opd * phase
        actual = np.dot(mat, opd) if mat.ndim > 1 else mat * opd
        
        if np.allclose(actual, expected, atol=1e-4):
            preserved_rots.append(rot)
            preserved_trans.append(trans)
    
    if len(preserved_rots) == 0:
        return 1, "P1"
    
    # Now apply the distortion to the supercell
    # The distortion pattern is: u(R) = eta * exp(i*q*R) * polarization
    # For simplicity, we apply a small distortion along the OPD direction
    
    # Get displacement pattern from irrep (this is simplified)
    # In practice, we'd need the full eigenvector
    displacement_amplitude = 0.1  # Small distortion
    
    # Apply distortion to each atom
    distorted_positions = []
    for i, pos in enumerate(sc_positions):
        # Get the lattice translation for this atom
        atom_idx = i % len(positions)
        cell_idx = i // len(positions)
        n = lattice_points[cell_idx][0]
        
        # Phase factor for this cell
        phase = np.exp(2j * np.pi * np.dot(qpoint, n))
        
        # Simplified: apply displacement along OPD direction
        # In reality, this should come from the phonon eigenvector
        displacement = displacement_amplitude * opd[0] * phase
        
        # Convert to Cartesian and apply (simplified)
        # This is a placeholder - real implementation needs full eigenvector
        dist_pos = pos.copy()
        dist_pos[0] += np.real(displacement) * 0.01  # Small perturbation
        dist_pos = dist_pos % 1.0
        distorted_positions.append(dist_pos)
    
    distorted_positions = np.array(distorted_positions)
    
    # Use spglib to identify the space group of the distorted structure
    distorted_cell = (sc_lattice, distorted_positions, sc_numbers)
    dataset = spglib.get_symmetry_dataset(distorted_cell, symprec=symprec)
    
    if dataset is None:
        # Try with standard primitive cell detection
        std_cell = spglib.standardize_cell(distorted_cell, symprec=symprec, to_primitive=False, no_idealize=False)
        if std_cell is not None:
            std_lattice, std_positions, std_numbers = std_cell
            dataset = spglib.get_symmetry_dataset((std_lattice, std_positions, std_numbers), symprec=symprec)
    
    if dataset is not None:
        return dataset.number, dataset.international
    
    return 0, "Unknown"


def compare_isotropy_methods(
    parent_lattice: np.ndarray,
    parent_rotations: np.ndarray,
    parent_translations: np.ndarray,
    qpoint: np.ndarray,
    irrep_matrices: np.ndarray,
    opd: np.ndarray,
    cell: Tuple[np.ndarray, np.ndarray, np.ndarray],
    symprec: float = 1e-5,
) -> Tuple[Tuple[int, str], Tuple[int, str], str]:
    """
    Compare the two methods for finding the daughter space group.
    
    Parameters:
        Same as get_isotropy_subgroup plus cell tuple
        
    Returns:
        (method1_result, method2_result, comparison_message)
    """
    # Method 1: Symmetry-based (current)
    sg_num_1, sg_sym_1 = get_isotropy_subgroup(
        parent_lattice, parent_rotations, parent_translations,
        qpoint, irrep_matrices, opd, symprec
    )
    
    # Method 2: Supercell-based (alternative)
    sg_num_2, sg_sym_2 = get_isotropy_subgroup_from_supercell(
        cell, qpoint, opd, irrep_matrices, symprec
    )
    
    # Compare
    if sg_num_1 == sg_num_2:
        msg = f"Both methods agree: {sg_sym_1} (#{sg_num_1})"
    else:
        msg = f"DISCREPANCY: Method 1={sg_sym_1} (#{sg_num_1}), Method 2={sg_sym_2} (#{sg_num_2})"
    
    return (sg_num_1, sg_sym_1), (sg_num_2, sg_sym_2), msg
