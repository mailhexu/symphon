import numpy as np
import spglib
from typing import Tuple
from itertools import product


def identify_spacegroup_from_operations(
    lattice: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    symprec: float = 1e-5,
    use_supercell_type: bool = False,
) -> Tuple[int, str]:
    """
    Identify the space group corresponding to a set of symmetry operations.
    
    Parameters:
        use_supercell_type: If True, use get_spacegroup_type_from_symmetry to
                           identify the SG without reducing to primitive cell.
    """
    if use_supercell_type:
        sg_type = spglib.get_spacegroup_type_from_symmetry(
            rotations, translations, lattice=lattice, symprec=symprec
        )
        if sg_type is None:
            return 0, "Unknown"
        
        num = sg_type.number if hasattr(sg_type, 'number') else sg_type.get('number', 0)
        sym = sg_type.international_short if hasattr(sg_type, 'international_short') else sg_type.get('international_short', '')
        
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
                t_prime = np.dot(S_inv, t + n) % 1.0
                # De-duplicate: skip if this (rotation, translation) pair already exists
                is_dup = any(
                    np.allclose(r_prime, sr) and np.allclose(t_prime, st)
                    for sr, st in zip(sc_rots, sc_trans)
                )
                if not is_dup:
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
    )
    if return_indices:
        return sg_num, sg_sym, preserved_indices
    return sg_num, sg_sym
