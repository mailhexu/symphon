import numpy as np
import spglib
from typing import Optional, List, Any, Dict, Tuple
from irreptables.irreps import IrrepTable
from spgrep import get_spacegroup_irreps_from_primitive_symmetry

def prepare_little_group_ops(
    primitive,
    qpoint,
    sg_symmetries,
    sg_metadata,
    table_symmetries: Optional[List[Any]] = None,
    symprec: float = 1e-5,
    log_level: int = 0
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    """
    Prepare little group operations with atom mapping and frame transformations.
    
    Returns:
        (sym_ops_processed, little_group_indices, mapping_to_table)
    """
    num_atoms = len(primitive.scaled_positions)
    positions = primitive.scaled_positions
    L = primitive.cell
    Linv = np.linalg.inv(L)
    
    is_gamma = (np.abs(qpoint) < symprec).all()
    refUC = sg_metadata.refUC
    shiftUC = sg_metadata.shiftUC
    refUCinv = np.linalg.inv(refUC)
    
    if is_gamma:
        q_work = qpoint
    else:
        q_work = refUC.T @ qpoint

    little_group_indices = []
    for i_sym, sym in enumerate(sg_symmetries):
        rot_bcs_lg = np.round(refUCinv @ sym.rotation @ refUC).astype(int)
        diff_bcs = rot_bcs_lg @ q_work - q_work
        if np.allclose(diff_bcs - np.round(diff_bcs), 0, atol=symprec):
            little_group_indices.append(i_sym)
            
    sym_ops_processed = []
    mapping_to_table = []
    
    table_syms_data = []
    if table_symmetries is not None:
        for s in table_symmetries:
            table_syms_data.append((s.R, s.t % 1.0))

    for idx, isym in enumerate(little_group_indices):
        sym = sg_symmetries[isym]
        rot_prim = sym.rotation
        trans_prim = sym.translation
        
        rot_bcs = np.round(refUCinv @ rot_prim @ refUC).astype(int)
        trans_bcs = refUCinv @ (trans_prim + rot_prim @ shiftUC - shiftUC)
        trans_bcs = np.round(trans_bcs, 10) % 1.0
        
        it_tab = -1
        if table_symmetries is not None:
            for i_tab, (R_tab, t_tab) in enumerate(table_syms_data):
                if np.array_equal(rot_bcs, R_tab):
                    dt = trans_bcs - t_tab
                    dt = dt - np.round(dt)
                    if np.allclose(dt, 0, atol=symprec):
                        it_tab = i_tab
                        break
        mapping_to_table.append(it_tab)

        R_cart = L @ rot_prim @ Linv
        
        perm = []
        phases = []
        rot_prim_inv = np.linalg.inv(rot_prim)
        for k in range(num_atoms):
            new_pos = rot_prim @ positions[k] + trans_prim
            found = False
            for j in range(num_atoms):
                diff = new_pos - positions[j]
                diff_round = np.round(diff)
                if np.allclose(diff - diff_round, 0, atol=symprec):
                    perm.append(j)
                    L_vec = rot_prim_inv @ (positions[j] - trans_prim) - positions[j]
                    phase = np.exp(2j * np.pi * np.dot(qpoint, L_vec))
                    phases.append(phase)
                    found = True
                    break
            if not found:
                raise RuntimeError(f"Atom mapping failed for sym {isym}")
        
        sym_ops_processed.append({
            'R_cart': R_cart, 
            'perm': perm, 
            'phases': phases,
            'rot_bcs': rot_bcs,
            'trans_bcs': trans_bcs
        })
        
    return sym_ops_processed, little_group_indices, mapping_to_table

def _match_symop_to_sg(
    rot: np.ndarray, trans: np.ndarray, sg_metadata: Any, symprec: float = 1e-5
) -> Optional[int]:
    for i, sym in enumerate(sg_metadata.symmetries):
        if np.array_equal(sym.rotation, rot):
            diff_t = sym.translation - trans
            diff_t = diff_t - np.round(diff_t)
            if np.allclose(diff_t, 0, atol=symprec):
                return i
    return None


def _get_spgrep_irreps_with_fallback(
    sg_metadata: Any,
    qpoint: np.ndarray,
    symprec: float = 1e-5,
    log_level: int = 0,
) -> Tuple[Optional[list], Optional[List[int]]]:
    """Try to get spgrep irreps, falling back to structure-based method.

    Returns (spgrep_irreps_list, spgrep_lg_indices) where spgrep_lg_indices
    are indices into sg_metadata.symmetries, or (None, None).
    """
    rotations = np.array([sym.rotation for sym in sg_metadata.symmetries], dtype=int)
    translations = np.array([sym.translation for sym in sg_metadata.symmetries], dtype=float)

    try:
        from spgrep import get_spacegroup_irreps_from_primitive_symmetry
        result = get_spacegroup_irreps_from_primitive_symmetry(
            rotations, translations, qpoint
        )
        spgrep_irreps_list = result[0] if isinstance(result, tuple) else result

        from spgrep.group import get_little_group
        _, _, spgrep_lg_indices = get_little_group(rotations, translations, qpoint, symprec)
        return spgrep_irreps_list, list(spgrep_lg_indices)
    except Exception as e:
        if log_level > 1:
            print(f"  spgrep primitive symmetry failed: {e}")

    try:
        lattice = getattr(sg_metadata, 'real_lattice', None)
        positions = getattr(sg_metadata, 'positions', None)
        typat = getattr(sg_metadata, 'typat', None)
        if lattice is None or positions is None or typat is None:
            if log_level > 1:
                print("  No cell data for structure-based fallback")
            return None, None

        from spgrep import get_spacegroup_irreps
        numbers = np.array(typat, dtype=int)
        irreps, sp_sg_rots, sp_sg_trans, lg_to_sg = get_spacegroup_irreps(
            lattice, positions, numbers, qpoint, method='random', symprec=symprec,
        )

        sp_lg_rots = sp_sg_rots[lg_to_sg]
        sp_lg_trans = sp_sg_trans[lg_to_sg]

        sg_lg_indices = []
        for rot, trans in zip(sp_lg_rots, sp_lg_trans):
            idx = _match_symop_to_sg(rot, trans, sg_metadata, symprec)
            if idx is None:
                if log_level > 1:
                    print(f"  Could not match spgrep op (rot={rot.tolist()}) to sg_metadata")
                return None, None
            sg_lg_indices.append(idx)

        reordered_irreps = []
        for irrep_mats in irreps:
            reordered = np.array([irrep_mats[i] for i in range(len(lg_to_sg))])
            reordered_irreps.append(reordered)

        if log_level > 1:
            print(f"  Structure-based fallback: {len(reordered_irreps)} irreps, lg_indices={sg_lg_indices}")
        return reordered_irreps, sg_lg_indices
    except Exception as e2:
        if log_level > 1:
            print(f"  Structure-based fallback failed: {e2}")

    return None, None


def get_reference_matrices(
    sg_metadata: Any, 
    label: str, 
    qpoint: np.ndarray, 
    bcs_kpname: Optional[str] = None,
    symprec: float = 1e-5,
    log_level: int = 0
) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
    """Get reference irrep matrices D(g) from spgrep matching the BCS label."""
    spgrep_irreps_list, spgrep_lg_indices = _get_spgrep_irreps_with_fallback(
        sg_metadata, qpoint, symprec, log_level
    )
    if spgrep_irreps_list is None or spgrep_lg_indices is None:
        return None, None

    n_sp_lg = len(spgrep_lg_indices)

    kpname = bcs_kpname or "GM"
    bcs_table = sg_metadata.get_irreps_from_table(kpname, qpoint)
    target_chars_orig = bcs_table.get(label)
    if target_chars_orig is None:
        return None, None

    target_char_vec = np.array(
        [target_chars_orig.get(int(idx) + 1, 0) for idx in spgrep_lg_indices],
        dtype=complex,
    )

    best_irrep_mats = None
    best_match = 0

    for irrep_mats in spgrep_irreps_list:
        chars = np.trace(irrep_mats, axis1=1, axis2=2)
        overlap = np.abs(np.vdot(target_char_vec, chars)) / n_sp_lg
        if overlap > best_match and overlap > 0.5:
            best_match = overlap
            best_irrep_mats = irrep_mats

    return best_irrep_mats, list(spgrep_lg_indices)

def get_combined_reference_matrices(
    sg_metadata: Any, 
    labels: List[str], 
    qpoint: np.ndarray,
    bcs_kpname: Optional[str] = None,
    symprec: float = 1e-5,
    log_level: int = 0
) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
    """Build combined reference matrices for a list of irrep labels."""
    all_refs = []
    shared_lg_indices = None
    for label in labels:
        ref, sp_lg = get_reference_matrices(sg_metadata, label, qpoint, bcs_kpname, symprec, log_level)
        if ref is None:
            return None, None
        if shared_lg_indices is None:
            shared_lg_indices = sp_lg
        all_refs.append(ref)

    n_lg = all_refs[0].shape[0]
    total_dim = sum(r.shape[1] for r in all_refs)
    combined = np.zeros((n_lg, total_dim, total_dim), dtype=complex)
    curr = 0
    for r in all_refs:
        d = r.shape[1]
        combined[:, curr:curr+d, curr:curr+d] = r
        curr += d
    return combined, shared_lg_indices

def solve_unitary_mapping(D: np.ndarray, M: np.ndarray) -> Optional[np.ndarray]:
    """
    Solve D(g) U = U M(g) for the unitary matrix U.
    
    Args:
        D: Reference IR matrices (order, dim, dim)
        M: Calculated representation matrices (order, dim, dim)
        
    Returns:
        Unitary matrix U such that D = U M U^dagger, or None if failed.
    """
    order = len(D)
    dim = D.shape[1]
    
    # U = 1/g * sum_g D(g) X M(g)^\dagger
    np.random.seed(42)
    X = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    U = np.zeros((dim, dim), dtype=complex)
    for g in range(order):
        U += D[g] @ X @ M[g].conj().T
    U /= order
    
    u_svd, s_svd, vh_svd = np.linalg.svd(U)
    if np.any(s_svd < 1e-4):
        return None
        
    return u_svd @ vh_svd

def calculate_phonon_representation_matrices(
    primitive,
    qpoint,
    little_group_ops: List[Dict[str, Any]],
    degenerate_sets: List[Tuple[int, ...]],
    eigvecs: np.ndarray,
    phase_convention: str = 'r',
    use_bcs_frame: bool = False,
    sg_metadata: Optional[Any] = None
) -> List[np.ndarray]:
    """
    Calculate representation matrices D(g) for each phonon degenerate block.
    
    Args:
        primitive: PhonopyAtoms object
        qpoint: q-point in primitive reciprocal basis
        little_group_ops: List of processed little group operations
        degenerate_sets: List of mode indices in degenerate blocks
        eigvecs: (3N, 3N) eigenvector matrix
        phase_convention: 'r' or 'R'
        use_bcs_frame: Whether to use BCS frame for phases (True for non-Gamma)
        sg_metadata: Optional SpaceGroupIrreps metadata for frame transform
        
    Returns:
        List of (num_little, dim, dim) matrices, one per degenerate block.
    """
    num_atoms = len(primitive.scaled_positions)
    positions = primitive.scaled_positions
    num_little = len(little_group_ops)
    
    # 1. Gauge transformation to 'r'-gauge if needed
    # Phonopy is 'R'-gauge (v). 'r'-gauge is w = v * exp(-i q tau)
    if use_bcs_frame and sg_metadata:
        refUC = sg_metadata.refUC
        shiftUC = sg_metadata.shiftUC
        refUCinv = np.linalg.inv(refUC)
        q_work = refUC.T @ qpoint
        positions_bcs = np.array([(refUCinv @ (p - shiftUC)) for p in positions])
        positions_bcs_mod = positions_bcs % 1.0
    else:
        q_work = qpoint
        positions_bcs_mod = positions
    
    all_block_matrices = []
    for block in degenerate_sets:
        dim = len(block)
        block_mats = np.zeros((num_little, dim, dim), dtype=complex)
        evs = eigvecs[:, block]
        
        if phase_convention == 'r':
            w_work = np.zeros((num_atoms, 3, dim), dtype=complex)
            evs_reshaped = evs.reshape(num_atoms, 3, dim)
            for k in range(num_atoms):
                pos_for_phase = positions_bcs_mod[k] if use_bcs_frame else positions[k]
                phase_w = np.exp(-2j * np.pi * np.dot(q_work, pos_for_phase))
                for d in range(dim):
                    w_work[k, :, d] = evs_reshaped[k, :, d] * phase_w
        else:
            w_work = evs.reshape(num_atoms, 3, dim)

        for i_lg, op in enumerate(little_group_ops):
            R_cart = op['R_cart']
            perm = op['perm']
            
            if phase_convention == 'r':
                q_prime = op.get('rot_bcs', np.eye(3)) @ q_work
                global_phase = np.exp(-2j * np.pi * np.dot(q_prime, op.get('trans_bcs', np.zeros(3))))
                G = q_prime - q_work
            else:
                global_phase = 1.0
                G = np.zeros(3)

            for m in range(dim):
                for n in range(dim):
                    val = 0
                    for k in range(num_atoms):
                        j = perm[k]
                        if phase_convention == 'r':
                            pos_j = positions_bcs_mod[j] if use_bcs_frame else positions[j]
                            phase_site = np.exp(2j * np.pi * np.dot(G, pos_j))
                            val += global_phase * phase_site * np.dot(w_work[j, :, m].conj(), R_cart @ w_work[k, :, n])
                        else:
                            val += op['phases'][k] * np.dot(w_work[j, :, m].conj(), R_cart @ w_work[k, :, n])
                    block_mats[i_lg, m, n] = val
        all_block_matrices.append(block_mats)
        
    return all_block_matrices

def match_block_to_table(
    irreps_in_table: List[Any],
    block_size: int,
    chars_calc: List[complex],
    lg_bcs_info: List[Tuple[int, complex]]
) -> Tuple[List[str], int]:
    """
    Match calculated traces to BCS irreps using the GOT formula.
    
    Args:
        irreps_in_table: List of Irrep objects from IrrepTable
        block_size: Dimension of the degenerate block
        chars_calc: Calculated characters (traces) for each little group op
        lg_bcs_info: List of (bcs_idx, phase_corr) for each little group op
        
    Returns:
        (matched_labels, total_irrep_dim)
    """
    num_little = len(chars_calc)
    matched_labels = []
    total_irrep_dim = 0

    for irr in irreps_in_table:
        if irr.dim > block_size:
            continue
        n_sgwv = len(irr.characters)
        if n_sgwv == 0:
            continue

        overlap = 0
        valid_match = True
        for idx in range(num_little):
            bcs_idx, phase_corr = lg_bcs_info[idx]
            if bcs_idx == -1:
                valid_match = False
                break
            if bcs_idx not in irr.characters:
                continue
            overlap += np.conj(irr.characters[bcs_idx]) * chars_calc[idx] * phase_corr

        if not valid_match:
            continue

        n = overlap / n_sgwv
        count = int(round(np.real(n)))
        if count > 0 and np.abs(n - count) < 0.2:
            for _ in range(count):
                matched_labels.append(irr.name)
                total_irrep_dim += irr.dim

    return matched_labels, total_irrep_dim
