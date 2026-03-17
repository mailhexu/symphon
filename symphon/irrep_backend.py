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
import sys
import os
import warnings


_context_manager = None
_redirected_stderr = False


def _suppress_spglib_warnings():
    """
    Suppress warnings from spglib C library when the little group
    operations don't form a recognizable point group.

    These warnings are informational (not errors) and printed
    directly to stderr by the C library. They can
    be suppressed by redirecting stderr to /dev/null before
    calling spglib functions that may trigger warnings.
    """
    # Save original stderr
    _original_stderr = sys.stderr
    
    # Redirect stderr to /dev/null
    sys.stderr = open(os.devnull, 'w')
    _redirected_stderr = True
     except Exception:
        _redirected_stderr = False
        raise RuntimeError("Failed to redirect stderr for spglib warning suppression")
     finally:
        if _redirected_stderr:
            sys.stderr = _original_stderr
            _redirected_stderr = False


def _restore_stderr():
    """Restore original stderr after processing."""
    if _redirected_stderr:
        sys.stderr = _original_stderr
        _redirected_stderr = False


def at with _suppress_spglib_warnings():
    rotations = np.array([sym.rotation for sym in sg.symmetries], dtype=int)
    with _suppress_spglib_warnings():
        pg_result = spglib.get_pointgroup(rotations)
    _restore_stderr()
    
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
        table = IrrepTable(str(self._spacegroup_number), sg.spinor)
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
        # 5. Try different phase conventions to find the best match
        best_irreps = None
        best_count = -1
        best_conv = self._phase_convention
        
        # Test both 'r' and 'R' gauges
        test_conventions = ['r', 'R']
        # If one was already specified, try it first
        if self._phase_convention in test_conventions:
            test_conventions.remove(self._phase_convention)
            test_conventions.insert(0, self._phase_convention)
            
        # Get irreps from the table for character matching
        irreps_in_table = [irr for irr in table.irreps if irr.kpname == kpname]
        
        for conv in test_conventions:
            self._phase_convention = conv
            if self._log_level > 1:
                print(f"\nTesting phase convention: '{conv}'")
            
            # 5.1 Calculate phonon representation matrices for each block
            block_matrices, little_group_indices, mapping_to_table = self._calculate_phonon_representations(sg, table=table)
            
            # 5.2 Match blocks with BCS labels
            current_irreps = []
            labeled_count = 0
            num_little = len(little_group_indices)

            for block_idx, block in enumerate(self._degenerate_sets):
                block_mats = block_matrices[block_idx]
                block_size = len(block)
                matched_labels = []
                total_irrep_dim = 0
                
                # Calculate characters for this block
                chars_calc = [np.trace(block_mats[i]) for i in range(num_little)]
                
                if self._log_level > 1:
                    print(f"    Block {block_idx} (dim {block_size}): matching with table...")

                # Match operations by rotation AND translation in BCS frame
                # Transform each sg.symmetries op to BCS frame and match to table
                refUC = sg.refUC
                shiftUC = sg.shiftUC
                refUCinv = np.linalg.inv(refUC)
                rot_trans_to_bcs_idx = {}
                for i_tab, sym_tab in enumerate(table.symmetries):
                    key = (tuple(sym_tab.R.flatten().tolist()), tuple((sym_tab.t % 1.0).tolist()))
                    rot_trans_to_bcs_idx[key] = i_tab + 1
                
                for irr in irreps_in_table:
                    # Allow matching composites if same dim
                    if irr.dim > block_size:
                        continue
                    
                    overlap = 0
                    valid_match = True
                    for idx, isym in enumerate(little_group_indices):
                        sym = sg.symmetries[isym]
                        rot_prim = sym.rotation
                        trans_prim = sym.translation
                        
                        # Transform to BCS frame
                        rot_bcs = np.round(refUCinv @ rot_prim @ refUC).astype(int)
                        trans_bcs = np.round(refUCinv @ (trans_prim + rot_prim @ shiftUC - shiftUC), 10) % 1.0
                        
                        key = (tuple(rot_bcs.flatten().tolist()), tuple(trans_bcs.tolist()))
                        bcs_idx = rot_trans_to_bcs_idx.get(key, -1)
                        if bcs_idx == -1:
                            valid_match = False
                            break
                        val = chars_calc[idx]
                        table_char = irr.characters.get(bcs_idx, 0)
                        overlap += np.conj(table_char) * val
                    
                    if not valid_match:
                        continue
                        
                    n = overlap / num_little
                    # n should be an integer (how many times this irrep is in the block)
                    count = int(round(np.real(n)))
                    if count > 0 and np.abs(n - count) < 0.2:
                        for _ in range(count):
                            matched_labels.append(irr.name)
                            total_irrep_dim += irr.dim
                    
                    if self._log_level > 1 and irr.kpname == kpname:
                        print(f"      - {irr.name}: match_val={np.abs(n):.4f} (overlap={overlap:.3f}, g={num_little})")

                # spgrep fallback
                if not matched_labels and HAS_SPGREP:
                    # Create a mini-bcs-table for label matching
                    mini_bcs_table = {irr.name: irr.characters for irr in irreps_in_table}
                    best_spgrep_label = self._label_block_with_spgrep(sg, little_group_indices, block_mats, mini_bcs_table)
                    if best_spgrep_label:
                        matched_labels = [best_spgrep_label]
                        total_irrep_dim = block_size
                
                # Store results for this gauge
                opds = [None] * block_size
                isotropy_sgs = ["-"] * block_size
                U = None
                
                if matched_labels:
                    labeled_count += 1
                    # Build combined label
                    from collections import Counter
                    counts = Counter(matched_labels)
                    sorted_unique = sorted(counts.keys())
                    match_label = "+".join([f"{counts[l]}*{l}" if counts[l] > 1 else l for l in sorted_unique])
                    
                    # Identify OPD if labels found
                    if total_irrep_dim == block_size:
                        try:
                            # Use spgrep for reference matrices if possible
                            if len(matched_labels) == 1:
                                ref_mats = self._get_reference_matrices(sg, little_group_indices, matched_labels[0])
                            else:
                                ref_mats = self._get_combined_reference_matrices(sg, little_group_indices, matched_labels)
                            
                            if ref_mats is not None:
                                U = self._solve_unitary_mapping(ref_mats, block_mats)
                                
                                # Get little group operations for isotropy subgroup
                                from .symmetry_identification import get_isotropy_subgroup
                                parent_lattice = self._primitive.cell
                                little_rots = np.array([sg.symmetries[idx].rotation for idx in little_group_indices], dtype=int)
                                little_trans = np.array([sg.symmetries[idx].translation for idx in little_group_indices], dtype=float)
                                
                                irrep_dim = ref_mats.shape[1]
                                
                                if U is not None and irrep_dim == 1:
                                    # For 1D irreps: only one possible OPD direction
                                    unit_vec = np.array([1.0], dtype=complex)
                                    opds[0] = self._column_to_opd_symbolic(unit_vec)
                                    try:
                                        sg_num, sg_sym = get_isotropy_subgroup(
                                            parent_lattice,
                                            little_rots,
                                            little_trans,
                                            self._qpoint,
                                            ref_mats,
                                            unit_vec,
                                            symprec=self._symprec,
                                        )
                                        isotropy_sgs[0] = f"{sg_sym}(#{sg_num})"
                                    except Exception:
                                        pass
                                else:
                                    # For 2D+ irreps: use IsotropyEnumerator to find proper OPDs
                                    try:
                                        from spgrep_modulation.isotropy import IsotropyEnumerator
                                        enumerator = IsotropyEnumerator(
                                            little_rotations=little_rots,
                                            little_translations=little_trans,
                                            qpoint=self._qpoint,
                                            small_rep=ref_mats.astype(complex),
                                        )
                                        
                                        # Collect OPDs from all maximal isotropy subgroups
                                        all_opds = []
                                        for subgroup_i, indices in enumerate(enumerator.maximal_isotropy_subgroups):
                                            opds_enum = enumerator.order_parameter_directions[subgroup_i]
                                            for opd_vec in opds_enum:
                                                all_opds.append(opd_vec)
                                        
                                        # Assign OPDs to modes
                                        for i in range(min(block_size, len(all_opds))):
                                            opd_vec = all_opds[i]
                                            opds[i] = self._column_to_opd_symbolic(opd_vec)
                                            
                                            try:
                                                sg_num, sg_sym = get_isotropy_subgroup(
                                                    parent_lattice,
                                                    little_rots,
                                                    little_trans,
                                                    self._qpoint,
                                                    ref_mats,
                                                    opd_vec,
                                                    symprec=self._symprec,
                                                )
                                                isotropy_sgs[i] = f"{sg_sym}(#{sg_num})"
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                else:
                    match_label = None

                # For acoustic modes at Gamma (zero freq), no symmetry is broken.
                # The isotropy subgroup is the full parent space group.
                is_gamma = np.allclose(self._qpoint, 0.0, atol=1e-5)
                if is_gamma and block and all(abs(self._freqs[j]) < 0.1 for j in block):
                    parent_sg_str = f"{self._spacegroup_symbol}(#{self._spacegroup_number})"
                    isotropy_sgs = [parent_sg_str] * block_size

                for i in range(block_size):
                    item = {
                        "label": match_label,
                        "opd": opds[i] if opds[i] is not None else "-",
                        "opd_num": U[:, i] if U is not None else None,
                        "daughter_sg": isotropy_sgs[i]
                    }
                    current_irreps.append(item)
                    
            if labeled_count > best_count:
                best_count = labeled_count
                best_irreps = current_irreps
                best_conv = conv
            
            # If we labeled all blocks, stop searching
            if labeled_count == len(self._degenerate_sets):
                break
                
        self._irreps = best_irreps
        self._phase_convention = best_conv
        if self._log_level > 0:
            print(f"Final phase convention: '{best_conv}' ({best_count}/{len(self._degenerate_sets)} blocks labeled)")
            
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
            spgrep_lg_ops = {
                i_sp: (rotations[idx_sp], translations[idx_sp])
                for i_sp, idx_sp in enumerate(spgrep_lg_indices)
            }
            for i_ph, idx_orig in enumerate(little_group_indices):
                rot_ph = rotations[idx_orig]
                trans_ph = translations[idx_orig]
                found = False
                for i_sp, (sp_rot, sp_trans) in spgrep_lg_ops.items():
                    if np.array_equal(rot_ph, sp_rot) and np.allclose(trans_ph % 1.0, sp_trans % 1.0, atol=1e-5):
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
        
    def _get_combined_reference_matrices(self, sg, little_group_indices, labels):
        """Build combined reference matrices for a list of irrep labels (reducible block)."""
        all_refs = []
        for label in labels:
            ref = self._get_reference_matrices(sg, little_group_indices, label)
            if ref is None:
                return None
            all_refs.append(ref)
            
        # all_refs is list of (n_lg, d_i, d_i)
        # We want (n_lg, total_dim, total_dim)
        n_lg = all_refs[0].shape[0]
        total_dim = sum(r.shape[1] for r in all_refs)
        combined = np.zeros((n_lg, total_dim, total_dim), dtype=complex)
        curr = 0
        for r in all_refs:
            d = r.shape[1]
            combined[:, curr:curr+d, curr:curr+d] = r
            curr += d
        return combined

    def _label_block_with_spgrep(self, sg, little_group_indices, block_matrices, bcs_table):
        """Use spgrep to identify the best label for a phonon block if manual matching fails."""
        if not HAS_SPGREP:
            return None
            
        # Calc characters of the block (traces of matrices)
        calc_chars = np.trace(block_matrices, axis1=1, axis2=2)
        n_lg = len(calc_chars)
        
        # Build character vectors from the BCS table for matching
        best_label = None
        max_overlap = 0
        
        bcs_indices = [idx + 1 for idx in little_group_indices]
        
        for label, table_chars in bcs_table.items():
            ref_chars = []
            for bcs_idx in bcs_indices:
                ref_chars.append(table_chars.get(bcs_idx, 0))
            ref_chars = np.array(ref_chars, dtype=complex)
            
            # Simple character overlap
            overlap = np.abs(np.vdot(ref_chars, calc_chars)) / n_lg
            
            # Divide by product of norms? No, n = 1/g sum chi_p chi_i*
            # multiplicity = overlap / dim_irrep? No, the trace of block is sum of traces.
            # So multiplicity = overlap.
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_label = label
                
        if max_overlap > 0.8:
            return best_label
        return None

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

    def _calculate_phonon_representations(self, sg, table=None):
        """Calculate representation matrices M(g) for each phonon block."""
        num_atoms = len(self._primitive.scaled_positions)
        positions = self._primitive.scaled_positions
        L = self._primitive.cell
        Linv = np.linalg.inv(L)
        q = self._qpoint
        
        # 1. Coordinate frame and origin shift
        is_gamma = (np.abs(self._qpoint) < self._symprec).all()
        refUC = sg.refUC
        shiftUC = sg.shiftUC
        refUCinv = np.linalg.inv(refUC)
        
        # BCS frame transformation for non-Gamma points
        # Used only for eigenvector transformation and character table matching
        if is_gamma:
            q_work = q
            use_bcs_frame = False
        else:
            q_work = refUC.T @ q
            use_bcs_frame = True

        # 3. Symmetry operations preparation
        little_group_indices = []
        for i_sym, sym in enumerate(sg.symmetries):
            # Check if this operation is a little group operation for q
            # Convention: R @ q = q (mod G), matching irrep package
            diff = (sym.rotation @ q - q)
            if np.allclose(diff - np.round(diff), 0, atol=self._symprec):
                little_group_indices.append(i_sym)
        
        num_little = len(little_group_indices)
        sym_ops = []
        mapping_to_table = []
        
        # Pre-process table symmetries for faster matching
        table_syms_data = []
        if table is not None:
            for s in table.symmetries:
                table_syms_data.append((s.R, s.t % 1.0))

        for idx, isym in enumerate(little_group_indices):
            sym = sg.symmetries[isym]
            rot_prim = sym.rotation
            trans_prim = sym.translation
            
            # Transform operation to BCS frame for table matching
            if use_bcs_frame:
                rot_bcs = np.round(refUCinv @ rot_prim @ refUC).astype(int)
                trans_bcs = refUCinv @ (trans_prim + rot_prim @ shiftUC - shiftUC)
                trans_bcs = np.round(trans_bcs, 10) % 1.0
            else:
                rot_bcs = rot_prim
                trans_bcs = np.round(trans_prim, 10) % 1.0
            
            # Find matching index in table (use BCS frame operations)
            it_tab = -1
            if table is not None:
                for i_tab, (R_tab, t_tab) in enumerate(table_syms_data):
                    if np.array_equal(rot_bcs, R_tab):
                        dt = trans_bcs - t_tab
                        dt = dt - np.round(dt)
                        if np.allclose(dt, 0, atol=self._symprec):
                            it_tab = i_tab
                            break
            mapping_to_table.append(it_tab)

            if self._log_level > 1 and idx < 8:
                if it_tab != -1:
                    print(f"  Debug: op {isym+1} matches table op {it_tab+1}")
                else:
                    print(f"  Debug: op {isym+1} NO MATCH in table! rot=\n{rot_bcs} trans={trans_bcs}")

            # Use phonopy's convention: R_cart = L @ rot @ inv(L)
            R_cart = L @ rot_prim @ Linv
            
            # Atom mapping in PRIMITIVE frame (always works)
            perm = []
            phases = []
            rot_prim_inv = np.linalg.inv(rot_prim)
            for k in range(num_atoms):
                new_pos = rot_prim @ positions[k] + trans_prim
                found = False
                for j in range(num_atoms):
                    diff = new_pos - positions[j]
                    diff_round = np.round(diff)
                    if np.allclose(diff - diff_round, 0, atol=self._symprec):
                        perm.append(j)
                        # Phase: exp(2πi * q @ (R^{-1} @ (τ_j - t) - τ_j))
                        L_vec = rot_prim_inv @ (positions[j] - trans_prim) - positions[j]
                        phase = np.exp(2j * np.pi * np.dot(q, L_vec))
                        phases.append(phase)
                        found = True
                        break
                if not found:
                    raise RuntimeError(f"Atom mapping failed for sym {isym}")
            
            sym_ops.append({
                'R_cart': R_cart, 
                'perm': perm, 
                'phases': phases,
                'rot_bcs': rot_bcs,
                'trans_bcs': trans_bcs
            })

        # 4. Block matrix calculation
        all_block_matrices = []
        for block in self._degenerate_sets:
            dim = len(block)
            block_mats = np.zeros((num_little, dim, dim), dtype=complex)
            evs = self._eigvecs[:, block]
            
            # Gauge transformation to 'r'-gauge if needed
            # Phonopy is 'R'-gauge (v). 'r'-gauge is w = v * exp(-i q tau)
            if use_bcs_frame:
                positions_bcs = np.array([(refUCinv @ (p - shiftUC)) for p in positions])
                positions_bcs_mod = positions_bcs % 1.0
            
            if self._phase_convention == 'r':
                w_work = np.zeros((num_atoms, 3, dim), dtype=complex)
                evs_reshaped = evs.reshape(num_atoms, 3, dim)
                for k in range(num_atoms):
                    pos_for_phase = positions_bcs_mod[k] if use_bcs_frame else positions[k]
                    phase_w = np.exp(-2j * np.pi * np.dot(q_work, pos_for_phase))
                    for d in range(dim):
                        vec = refUCinv @ evs_reshaped[k, :, d] if use_bcs_frame else evs_reshaped[k, :, d]
                        w_work[k, :, d] = vec * phase_w
            else:
                # Keep 'R'-gauge (v)
                w_work = np.zeros((num_atoms, 3, dim), dtype=complex)
                evs_reshaped = evs.reshape(num_atoms, 3, dim)
                for k in range(num_atoms):
                    for d in range(dim):
                        vec = refUCinv @ evs_reshaped[k, :, d] if use_bcs_frame else evs_reshaped[k, :, d]
                        w_work[k, :, d] = vec

            for i_lg in range(num_little):
                op = sym_ops[i_lg]
                R_cart = op['R_cart']
                perm = op['perm']
                
                if self._phase_convention == 'r':
                    # 'r'-gauge: D(g) = exp(-i Rq t) sum_k R exp(i G tau_j) delta_j,perm(k)
                    q_prime = op['rot_bcs'] @ q_work
                    global_phase = np.exp(-2j * np.pi * np.dot(q_prime, op['trans_bcs']))
                    G = q_prime - q_work
                else:
                    global_phase = 1.0

                for m in range(dim):
                    for n in range(dim):
                        val = 0
                        for k in range(num_atoms):
                            j = perm[k]
                            if self._phase_convention == 'r':
                                pos_j = positions_bcs_mod[j] if use_bcs_frame else positions[j]
                                phase_site = np.exp(2j * np.pi * np.dot(G, pos_j))
                                val += global_phase * phase_site * np.dot(w_work[j, :, m].conj(), R_cart @ w_work[k, :, n])
                            else:
                                val += op['phases'][k] * np.dot(w_work[j, :, m].conj(), R_cart @ w_work[k, :, n])
                        block_mats[i_lg, m, n] = val
            all_block_matrices.append(block_mats)
            
        return all_block_matrices, little_group_indices, mapping_to_table
