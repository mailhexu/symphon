import numpy as np
from phonopy.phonon.degeneracy import degenerate_sets
from ase.data import atomic_masses
import logging
import spglib
from irrep.spacegroup_irreps import SpaceGroupIrreps
from spgrep import get_spacegroup_irreps_from_primitive_symmetry

from .chiral_transitions import opd_to_symbolic, HAS_SPGREP

class IrRepsIrrep:
    """
    Backend driver using the 'irrep' package for irrep identification.
    """
    def __init__(self, primitive, qpoint, freqs, eigvecs, symprec=1e-5, log_level=0, phase_convention='r'):
        self._primitive = primitive # PhonopyAtoms
        self._qpoint = np.array(qpoint)
        self._freqs = freqs
        self._eigvecs = eigvecs # (3N, 3N)
        self._symprec = symprec
        self._log_level = log_level
        self._phase_convention = phase_convention  # 'r' (default) or 'R'
        
        self._irreps = None
        self._degenerate_sets = None
        self._pointgroup_symbol = None
        self._spacegroup_symbol = None
        self._spacegroup_number = None
        self._RamanIR_labels = None # Not implemented for non-gamma in this backend yet
        self._bcs_kpname = None  # Store BCS k-point label
        self._qpoint_bcs = None  # Store k-point in BCS coordinates
        
    def run(self, kpname=None):
        # 1. Setup SpaceGroupIrreps from irrep package
        cell = (self._primitive.cell, self._primitive.scaled_positions, self._primitive.numbers)
        sg = SpaceGroupIrreps.from_cell(
            cell=cell,
            spinor=False,
            include_TR=False,
            search_cell=True,
            symprec=self._symprec,
            verbosity=self._log_level
        )
        self._sg_obj = sg
        from phonopy.structure.symmetry import Symmetry
        self._sg_phonopy = Symmetry(self._primitive, symprec=self._symprec).dataset
        
        self._spacegroup_symbol = sg.name
        self._spacegroup_number = int(sg.number_str.split('.')[0])
        self._degenerate_sets = degenerate_sets(self._freqs)
        
        # Get point group from symmetry operations
        rotations = np.array([sym.rotation for sym in sg.symmetries], dtype=int)
        pg_result = spglib.get_pointgroup(rotations)
        if pg_result:
            self._pointgroup_symbol = pg_result[0]
        else:
            self._pointgroup_symbol = None
        
        # 2. Get BCS character table for the q-point
        # Use refUC to find the matching label in the BCS table
        try:
            from irreptables.irreps import IrrepTable
        except ImportError:
            from irreptables import IrrepTable
        table = IrrepTable(sg.number_str, sg.spinor)
        refUC = sg.refUC
        refUCTinv = np.linalg.inv(refUC.T)
        
        # Transform q-point to BCS coordinates
        q_bcs = refUC.T @ self._qpoint
        self._qpoint_bcs = q_bcs
        
        found_kpname = None
        for irr in table.irreps:
            k_prim_table = (refUCTinv @ irr.k)
            diff = (k_prim_table - self._qpoint)
            if np.allclose(diff - np.round(diff), 0, atol=1e-4):
                found_kpname = irr.kpname
                break
        
        # Print coordinate mapping information
        if self._log_level > 0:
            print(f"\nK-point Coordinate Mapping:")
            print(f"  Primitive: [{self._qpoint[0]:.4f}, {self._qpoint[1]:.4f}, {self._qpoint[2]:.4f}]")
            print(f"  BCS:       [{q_bcs[0]:.4f}, {q_bcs[1]:.4f}, {q_bcs[2]:.4f}]")
            if found_kpname:
                print(f"  BCS Label: {found_kpname}")
        
        if kpname is not None and found_kpname is not None and kpname != found_kpname:
            if self._log_level > 0:
                print(f"Warning: Provided kpname '{kpname}' differs from detected BCS label '{found_kpname}'. Using '{found_kpname}'.")
        
        if found_kpname:
            kpname = found_kpname
            self._bcs_kpname = found_kpname
        elif kpname is None:
            # Fallback to Gamma if close to zero
            if (np.abs(self._qpoint) < self._symprec).all():
                kpname = "GM"
            else:
                raise ValueError(f"Could not identify BCS label for q-point {self._qpoint}. Please provide kpname.")
        
        try:
            # Pass original qpoint, irrep handles transformation internally
            bcs_table = sg.get_irreps_from_table(kpname, self._qpoint)
        except Exception as e:
            if self._log_level > 0:
                print(f"Error fetching BCS table for {kpname} at {self._qpoint}: {e}")
            raise e

        # 3. Detect if BCS table has only multi-dimensional irreps
        # Check dimension of each irrep (identity character = dimension)
        min_irrep_dim = float('inf')
        for label, table_chars in bcs_table.items():
            if 1 in table_chars:
                identity_char = table_chars[1]
            else:
                identity_char = list(table_chars.values())[0]
            dim = int(round(abs(identity_char)))
            min_irrep_dim = min(min_irrep_dim, dim)
        
        only_multidim = (min_irrep_dim > 1)
        
        if only_multidim and self._log_level > 0:
            print(f"\nDetected k-point with ONLY {min_irrep_dim}D irreps (no 1D irreps)")
            print(f"Force-pairing consecutive modes into {min_irrep_dim}D blocks...")
        
        if self._log_level > 0:
            print(f"Using phase convention: '{self._phase_convention}'-gauge")
        
        # 4. Apply force-pairing if needed
        if only_multidim and min_irrep_dim == 2:
            # Force pair consecutive modes for 2D irreps
            original_deg_sets = self._degenerate_sets
            forced_pairs = []
            i = 0
            while i < len(self._freqs):
                if i + 1 < len(self._freqs):
                    # Pair consecutive modes
                    forced_pairs.append(tuple([i, i+1]))
                    i += 2
                else:
                    # Odd number of modes - leave last one as single
                    forced_pairs.append(tuple([i]))
                    i += 1
            
            self._degenerate_sets = forced_pairs
            
            if self._log_level > 0:
                print(f"  Original degeneracy: {len(original_deg_sets)} blocks")
                print(f"  Forced pairing: {len(forced_pairs)} blocks")
                singles = sum(1 for b in forced_pairs if len(b) == 1)
                pairs = sum(1 for b in forced_pairs if len(b) == 2)
                print(f"    Singles: {singles}, Pairs: {pairs}")
        
        # 5. Calculate phonon representation matrices for each block
        self._block_matrices, self._little_group_indices = self._calculate_phonon_representations(sg)
        block_matrices = self._block_matrices
        little_group_indices = self._little_group_indices

        # 6. Match blocks with BCS labels and identify OPDs
        self._irreps = []
        for block_idx, block in enumerate(self._degenerate_sets):
            block_size = len(block)
            
            # Match with BCS characters first to get the label
            best_match_label = None
            best_irrep_dim = None
            max_overlap = 0
            
            # Map little_group_indices to BCS indices (1-based)
            bcs_indices = [idx + 1 for idx in little_group_indices]
            
            for label, table_chars in bcs_table.items():
                # Check irrep dimension from identity character
                identity_char = table_chars.get(1, list(table_chars.values())[0])
                irrep_dim = int(round(abs(identity_char)))
                
                g = len(table_chars)
                overlap = 0
                for i_lg, i_bcs in enumerate(bcs_indices):
                    if i_bcs in table_chars:
                        # Tr(M(g)) = sum of diagonal elements
                        tr_M = np.trace(block_matrices[block_idx][i_lg])
                        overlap += np.conj(table_chars[i_bcs]) * tr_M
                
                n = overlap / g
                match_val = np.abs(n)
                
                if match_val > max_overlap:
                    max_overlap = match_val
                    best_match_label = label
                    best_irrep_dim = irrep_dim
            
            # 7. Identify OPD if label found
            is_gamma = (np.abs(self._qpoint) < self._symprec).all()
            threshold = 0.8 if is_gamma else 0.5
            
            opds = [None] * block_size
            isotropy_sgs = ["-"] * block_size
            
            U = None
            if best_match_label and max_overlap > threshold:
                # Get reference matrices from spgrep for this irrep
                try:
                    ref_matrices = self._get_reference_matrices(sg, little_group_indices, best_match_label)
                    
                    if ref_matrices is not None and best_irrep_dim == block_size:
                        # Solve D(g) U = U M(g) for unitary mapping U
                        # Only possible when irrep dimension matches block size
                        U = self._solve_unitary_mapping(ref_matrices, block_matrices[block_idx])
                        
                        if U is not None:
                            # Rotate eigenvectors to match the standard irrep basis
                            self._eigvecs[:, block] = self._eigvecs[:, block] @ U.conj().T
                            for i in range(block_size):
                                unit_vec = np.zeros(block_size, dtype=complex)
                                unit_vec[i] = 1.0
                                opds[i] = self._column_to_opd_symbolic(unit_vec)
                                
                                # Identify daughter space group
                                try:
                                    from .symmetry_identification import get_isotropy_subgroup
                                    parent_lattice = self._primitive.cell
                                    # ref_matrices are indexed in sg.symmetries order.
                                    # Use sg.symmetries directly (same ordering as BCS/spgrep).
                                    little_rots = np.array([sg.symmetries[idx].rotation for idx in little_group_indices], dtype=int)
                                    little_trans = np.array([sg.symmetries[idx].translation for idx in little_group_indices], dtype=float)
                                    
                                    sg_num, sg_sym = get_isotropy_subgroup(
                                        parent_lattice,
                                        little_rots,
                                        little_trans,
                                        self._qpoint,
                                        ref_matrices,
                                        unit_vec,
                                        symprec=self._symprec,
                                    )
                                    isotropy_sgs[i] = f"{sg_sym}(#{sg_num})"
                                except Exception:
                                    isotropy_sgs[i] = "-"
                except Exception as e:
                    if self._log_level > 0:
                        print(f"  Debug: Backend failed for block {block}: {e}")

            # Always add entries for all modes in this block
            for i in range(block_size):
                item = {
                    "label": best_match_label if max_overlap > threshold else None,
                    "opd": opds[i] if opds[i] is not None else "-",
                    "opd_num": U[:, i] if U is not None else None,
                    "daughter_sg": isotropy_sgs[i]
                }
                self._irreps.append(item)
                    
        return True

    def _get_reference_matrices(self, sg, little_group_indices: list[int], label: str):
        """Get reference irrep matrices D(g) from spgrep matching the BCS label."""
        if not HAS_SPGREP:
            return None
            
        # 1. Get all irreps from spgrep at this q-point
        rotations = np.array([sym.rotation for sym in sg.symmetries], dtype=int)
        translations = np.array([sym.translation for sym in sg.symmetries], dtype=float)
        
        try:
            result = get_spacegroup_irreps_from_primitive_symmetry(
                rotations, translations, self._qpoint
            )
            # spgrep returns (list_of_irrep_matrices, indicators)
            if isinstance(result, tuple):
                spgrep_irreps_list = result[0]
            else:
                spgrep_irreps_list = result
        except Exception as e:
            if self._log_level > 1:
                print(f"  spgrep call failed: {e}")
            return None

        # 2. Find which spgrep irrep matches the BCS label by comparing characters
        #    IMPORTANT: spgrep returns irreps indexed by little-group operations
        #    (0..n_lg-1), NOT by the full space group. We need to map:
        #      spgrep index i  ↔  full SG index little_group_indices[i]  ↔  BCS index (SG_idx + 1)
        kpname = self._bcs_kpname or "GM"
        # The BCS table contains irreps for the operations of the space group.
        # We need to map Phonopy rotations (parent_rots) to BCS rotation indices.
        
        # 1. Get rotations from the BCS table for this space group
        try:
            from irreptables.irreps import IrrepTable
        except ImportError:
            from irreptables import IrrepTable  # type: ignore
        bcs_sg_obj = IrrepTable(str(self._spacegroup_number), spinor=False)
        
        # 2. Build mapping from rotation matrix to BCS index (1..nsym)
        def mat_to_tuple(m):
            return tuple(m.flatten().tolist())
        
        rot_to_bcs_idx = {}
        for i_bcs, sym_op in enumerate(bcs_sg_obj.symmetries):
            rot_to_bcs_idx[mat_to_tuple(sym_op.R)] = i_bcs + 1
            
        bcs_table = sg.get_irreps_from_table(kpname, self._qpoint)
        target_chars_orig = bcs_table.get(label)
        if target_chars_orig is None:
            return None
            
        # 3. Build target character vector for the little group.
        # little_group_indices are indices into sg.symmetries (the irrep package's list),
        # which has the SAME ordering as the BCS table (sg[i] => BCS index i+1).
        # So we look up target_chars_orig[idx+1] directly — no rotation matrix lookup needed.
        n_lg = len(little_group_indices)
        target_char_val_list = []
        for idx in little_group_indices:
            bcs_idx = idx + 1  # sg.symmetries[idx] == BCS op (idx+1)
            target_char_val_list.append(target_chars_orig.get(bcs_idx, 0))
        
        target_char_vec = np.array(target_char_val_list, dtype=complex)
        # rotations/translations are for full SG
        # little_group_indices are indices of Phonopy SG rots that fix q
        lg_rots_target = rotations[little_group_indices]
        
        best_irrep_mats = None
        best_match = 0
        
        for irrep_mats in spgrep_irreps_list:
            # irrep_mats is shaped (n_spgrep_lg, d, d)
            # Match them to lg_rots_target
            from spgrep.group import get_little_group
            _, _, spgrep_lg_indices = get_little_group(rotations, translations, self._qpoint, self._symprec)
            
            if len(spgrep_lg_indices) != len(little_group_indices):
                continue
                
            # Build mapping from little_group_indices (irrep/BCS ordering) to
            # spgrep_lg_indices (spgrep ordering). These use different orderings,
            # so we must match by rotation matrix values, not by index.
            phonopy_to_spgrep = {}
            spgrep_lg_rots = {i_sp: rotations[idx_sp] for i_sp, idx_sp in enumerate(spgrep_lg_indices)}
            for i_ph, idx_orig in enumerate(little_group_indices):
                rot_ph = rotations[idx_orig]
                found = False
                for i_sp, sp_rot in spgrep_lg_rots.items():
                    if np.array_equal(rot_ph, sp_rot):
                        phonopy_to_spgrep[i_ph] = i_sp
                        found = True
                        break
                if not found:
                    break
            
            if self._log_level > 1:
                print(f"  Debug: phonopy_to_spgrep={phonopy_to_spgrep}")
                print(f"  Debug: spgrep_lg_indices={spgrep_lg_indices}")
                
            lg_mats = np.array([irrep_mats[phonopy_to_spgrep[i]] for i in range(len(little_group_indices))])
            chars = np.trace(lg_mats, axis1=1, axis2=2)
            
            overlap = np.abs(np.vdot(target_char_vec, chars)) / n_lg
            if self._log_level > 1:
                print(f"  Debug: target_chars[:5]={target_char_vec[:5]}")
                print(f"  Debug: irrep overlap={overlap:.4f} chars[:5]={chars[:5]}")
            if overlap > best_match and overlap > 0.5:
                best_match = overlap
                best_irrep_mats = lg_mats
        
        if self._log_level > 1 and best_irrep_mats is not None:
            print(f"  Matched spgrep irrep for {label}: overlap={best_match:.4f}")
        
        return best_irrep_mats



    def _solve_unitary_mapping(self, D, M):
        """Solve D(g) U = U M(g) for the unitary matrix U."""
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

    @staticmethod
    def _column_to_opd_symbolic(col: np.ndarray, tol: float = 1e-3) -> str:
        """
        Convert a column of the unitary mapping U to a symbolic OPD string.

        Algorithm:
            1. Factor out the global phase so the largest-magnitude entry becomes
               real and positive.
            2. Round near-zero entries to exactly zero.
            3. Group nonzero entries by value equality (up to sign/factor of i).
            4. Assign symbolic parameters a, b, c, ...
        """
        dim = len(col)
        
        # 1. Factor out global phase: make largest entry real and positive
        max_idx = np.argmax(np.abs(col))
        phase = np.exp(-1j * np.angle(col[max_idx])) if np.abs(col[max_idx]) > tol else 1.0
        v = col * phase

        # 2. Clean near-zero imaginary/real parts
        for i in range(dim):
            if np.abs(v[i].imag) < tol:
                v[i] = complex(v[i].real, 0)
            if np.abs(v[i].real) < tol:
                v[i] = complex(0, v[i].imag)
            if np.abs(v[i]) < tol:
                v[i] = 0

        # 3. Group components by proportionality
        # For each nonzero entry, normalize to unit magnitude and
        # track the magnitude + relative sign/phase
        param_labels = [''] * dim
        next_param = 0
        param_chars = 'abcdefgh'
        assigned = [False] * dim

        for i in range(dim):
            if assigned[i]:
                continue
            if np.abs(v[i]) < tol:
                param_labels[i] = '0'
                assigned[i] = True
                continue

            # This is a new independent parameter
            char = param_chars[next_param] if next_param < len(param_chars) else f'p{next_param}'
            next_param += 1
            ref_val = v[i]
            param_labels[i] = char
            assigned[i] = True

            # Check remaining entries for proportionality
            for j in range(i + 1, dim):
                if assigned[j] or np.abs(v[j]) < tol:
                    continue
                ratio = v[j] / ref_val
                # Check if ratio is a simple real or imaginary factor
                if np.isclose(ratio, 1, atol=tol):
                    param_labels[j] = char
                    assigned[j] = True
                elif np.isclose(ratio, -1, atol=tol):
                    param_labels[j] = f'-{char}'
                    assigned[j] = True
                elif np.isclose(ratio, 1j, atol=tol):
                    param_labels[j] = f'i{char}'
                    assigned[j] = True
                elif np.isclose(ratio, -1j, atol=tol):
                    param_labels[j] = f'-i{char}'
                    assigned[j] = True
                # else: different magnitude/phase → will get own parameter

        # Mark any remaining unassigned nonzero entries
        for i in range(dim):
            if not assigned[i] and np.abs(v[i]) >= tol:
                char = param_chars[next_param] if next_param < len(param_chars) else f'p{next_param}'
                next_param += 1
                param_labels[i] = char
                assigned[i] = True

        return '(' + ', '.join(param_labels) + ')'

    def _calculate_phonon_representations(self, sg):
        """Calculate the full representation matrices M(g) for each degenerate block."""
        num_atoms = len(self._primitive.scaled_positions)
        positions = self._primitive.scaled_positions
        
        # Identify little group of q
        little_group_indices = []
        for i, sym in enumerate(sg.symmetries):
            dq = np.dot(sym.rotation, self._qpoint) - self._qpoint
            if np.allclose(dq - np.round(dq), 0, atol=1e-5):
                little_group_indices.append(i)
        
        is_gamma = (np.abs(self._qpoint) < self._symprec).all()
        refUC = sg.refUC
        shiftUC = sg.shiftUC
        refUCinv = np.linalg.inv(refUC)
        
        if is_gamma:
            positions_work = positions
            positions_work_unmodded = positions
            q_work = self._qpoint
            use_bcs_frame = False
        else:
            positions_bcs_unmodded = np.array([(refUCinv @ (p - shiftUC)) for p in positions])
            positions_work = positions_bcs_unmodded % 1
            positions_work_unmodded = positions_bcs_unmodded
            q_work = refUC.T @ self._qpoint
            use_bcs_frame = True
                
        num_little = len(little_group_indices)
        L = self._primitive.cell
        Linv = np.linalg.inv(L)
        
        if use_bcs_frame:
            L_bcs = L @ refUC
            L_bcs_inv = np.linalg.inv(L_bcs)
        else:
            L_bcs = L
            L_bcs_inv = Linv

        sym_ops = []
        for idx, isym in enumerate(little_group_indices):
            sym = sg.symmetries[isym]
            rot_prim = sym.rotation
            trans_prim = sym.translation
            
            if use_bcs_frame:
                rot_work = np.round(refUCinv @ rot_prim @ refUC).astype(int)
                trans_work = refUCinv @ (trans_prim + rot_prim @ shiftUC - shiftUC)
            else:
                rot_work = rot_prim
                trans_work = trans_prim
            
            R_cart = L_bcs.T @ rot_work @ L_bcs_inv.T
            
            perm = []
            phases = []
            for k in range(num_atoms):
                new_pos = rot_work @ positions_work[k] + trans_work
                found = False
                for j in range(num_atoms):
                    diff = new_pos - positions_work[j]
                    diff_round = np.round(diff)
                    if np.allclose(diff - diff_round, 0, atol=self._symprec):
                        perm.append(j)
                        if self._phase_convention == 'R':
                            phase = 1.0
                        else:
                            L_vec = rot_work @ positions_work_unmodded[k] + trans_work - positions_work_unmodded[k]
                            phase = np.exp(2j * np.pi * np.dot(q_work, L_vec))
                        phases.append(phase)
                        found = True
                        break
                if not found:
                    raise RuntimeError(f"Atom mapping failed for sym {sym.ind}")
            sym_ops.append({'R_cart': R_cart, 'perm': perm, 'phases': phases})

        all_block_matrices = []
        for block in self._degenerate_sets:
            dim = len(block)
            block_mats = np.zeros((num_little, dim, dim), dtype=complex)
            evs = self._eigvecs[:, block]
            if use_bcs_frame:
                evs_reshaped = evs.reshape(num_atoms, 3, dim)
                evs_work = np.zeros_like(evs_reshaped)
                for d in range(dim):
                    evs_work[:, :, d] = (refUCinv @ evs_reshaped[:, :, d].T).T
            else:
                evs_work = evs.reshape(num_atoms, 3, dim)

            for i_lg in range(num_little):
                op = sym_ops[i_lg]
                R_cart = op['R_cart']
                perm = op['perm']
                phases = op['phases']
                
                for m in range(dim):
                    for n in range(dim):
                        val = 0
                        for k in range(num_atoms):
                            j = perm[k]
                            val += phases[k] * np.dot(evs_work[j, :, m].conj(), R_cart @ evs_work[k, :, n])
                        block_mats[i_lg, m, n] = val
            all_block_matrices.append(block_mats)
            
        return all_block_matrices, little_group_indices
