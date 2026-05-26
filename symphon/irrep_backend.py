import numpy as np
from phonopy.phonon.degeneracy import degenerate_sets
from ase.data import atomic_masses
import logging
import sys
import os
from dataclasses import dataclass
from contextlib import contextmanager
import spglib
from irrep.spacegroup_irreps import SpaceGroupIrreps
from spgrep import get_spacegroup_irreps_from_primitive_symmetry

from .chiral_transitions import opd_to_symbolic, HAS_SPGREP


@contextmanager
def _suppress_spglib_warnings():
    """Suppress spglib C/Python warnings printed to stderr."""
    devnull = open(os.devnull, 'w')
    old_stderr = sys.stderr
    old_stderr_fd = os.dup(2)
    try:
        sys.stderr = devnull
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        sys.stderr = old_stderr
        devnull.close()


@dataclass
class BcsLabelContext:
    """BCS label data in the operation order used by SpaceGroupIrreps."""

    kpname: str
    qpoint: np.ndarray
    qpoint_bcs: np.ndarray
    adjusted_characters: dict[str, dict]
    table_irreps: list

    @classmethod
    def from_spacegroup(cls, sg, table, kpname: str, qpoint: np.ndarray):
        adjusted_characters = sg.get_irreps_from_table(kpname, qpoint)
        return cls(
            kpname=kpname,
            qpoint=qpoint,
            qpoint_bcs=sg.refUC.T @ qpoint,
            adjusted_characters=adjusted_characters,
            table_irreps=[irr for irr in table.irreps if irr.kpname == kpname],
        )

    def match_block(self, chars_calc, block_size: int, little_group_indices: list[int], symprec: float):
        """Return exact labels, total dimension, and heuristic fallback label."""
        matched_labels = []
        total_irrep_dim = 0
        match_scores = []

        for irr in self.table_irreps:
            if irr.dim > block_size:
                continue

            table_chars = self.adjusted_characters.get(irr.name)
            if not table_chars:
                continue

            overlap = 0
            for i_lg, isym in enumerate(little_group_indices):
                char = table_chars.get(int(isym) + 1)
                if char is None:
                    continue
                overlap += np.conj(char) * chars_calc[i_lg]

            n_sgwv = len(table_chars)
            if n_sgwv == 0:
                continue

            n = overlap / n_sgwv
            score = float(np.abs(n))
            match_scores.append((irr.name, irr.dim, score, n, overlap, n_sgwv))

            count = int(round(np.real(n)))
            if count > 0 and np.abs(n - count) < 0.2:
                for _ in range(count):
                    matched_labels.append(irr.name)
                    total_irrep_dim += irr.dim

        heuristic_label = None
        if not matched_labels and match_scores and not np.allclose(self.qpoint, 0.0, atol=symprec):
            sorted_scores = sorted(match_scores, key=lambda item: item[2], reverse=True)
            best_score = sorted_scores[0][2]
            if best_score >= 0.35:
                best_names = sorted([
                    name for name, _dim, score, _n, _overlap, _g in sorted_scores
                    if best_score - score <= 0.08
                ])
                heuristic_label = "/".join(best_names)

        return matched_labels, total_irrep_dim, heuristic_label, match_scores

class IrRepsIrrep:
    """
    Backend driver using the 'irrep' package for irrep identification.
    """
    def __init__(self, primitive, qpoint, freqs, eigvecs, symprec=1e-5, log_level=0, phase_convention='r', compute_heuristic_opds=False):
        self._primitive = primitive # PhonopyAtoms
        self._qpoint = np.array(qpoint)
        self._freqs = freqs
        self._eigvecs = eigvecs # (3N, 3N)
        self._symprec = symprec
        self._log_level = log_level
        self._phase_convention = phase_convention  # 'r' (default) or 'R'
        self._compute_heuristic_opds = compute_heuristic_opds
        
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
        with _suppress_spglib_warnings():
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
        
        rotations = np.array([sym.rotation for sym in sg.symmetries], dtype=int)
        with _suppress_spglib_warnings():
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
            label_context = BcsLabelContext.from_spacegroup(sg, table, kpname, self._qpoint)
            bcs_table = label_context.adjusted_characters
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
        
        # 4. Initialize degenerate sets from frequencies
        self._degenerate_sets = degenerate_sets(self._freqs)
        
        # 5. Apply force-pairing if needed
        # Force-pair only when ALL natural blocks are singletons (size=1) and we have
        # only multi-dim irreps.  When natural blocks already have size >= min_irrep_dim
        # (e.g. 4-fold degeneracy at P where P1 and P2 are both present), keep the
        # natural degenerate sets – splitting them would yield arbitrary mixtures.
        if only_multidim and min_irrep_dim == 2:
            original_deg_sets = self._degenerate_sets
            max_natural_block = max(len(b) for b in original_deg_sets)
            if max_natural_block < min_irrep_dim:
                # All blocks are singletons – force-pair consecutive modes
                forced_pairs = []
                i = 0
                while i < len(self._freqs):
                    if i + 1 < len(self._freqs):
                        forced_pairs.append(tuple([i, i+1]))
                        i += 2
                    else:
                        forced_pairs.append(tuple([i]))
                        i += 1
                self._degenerate_sets = forced_pairs
                if self._log_level > 0:
                    print(f"  Original degeneracy: {len(original_deg_sets)} blocks")
                    print(f"  Forced pairing: {len(forced_pairs)} blocks")
                    singles = sum(1 for b in forced_pairs if len(b) == 1)
                    pairs = sum(1 for b in forced_pairs if len(b) == 2)
                    print(f"    Singles: {singles}, Pairs: {pairs}")
            else:
                # Natural blocks already >= min_irrep_dim; keep them as-is
                if self._log_level > 0:
                    print(f"  Original degeneracy: {len(original_deg_sets)} blocks (size>={max_natural_block}; no force-pairing needed)")
        
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

            if self._log_level > 1:
                print(
                    f"  BCS context: kpname={kpname}, q_bcs={label_context.qpoint_bcs}, "
                    f"LG size={len(little_group_indices)}"
                )

            for block_idx, block in enumerate(self._degenerate_sets):
                block_mats = block_matrices[block_idx]
                block_size = len(block)
                matched_labels = []
                labels_for_opd = []
                heuristic_label = None
                total_irrep_dim = 0
                
                # Calculate characters for this block
                chars_calc = [np.trace(block_mats[i]) for i in range(num_little)]
                
                if self._log_level > 1:
                    print(f"    Block {block_idx} (dim {block_size}): matching with table...")

                matched_labels, total_irrep_dim, heuristic_label, match_scores = label_context.match_block(
                    chars_calc,
                    block_size,
                    little_group_indices,
                    self._symprec,
                )

                if self._log_level > 1:
                    for name, _dim, score, n, overlap, n_sgwv in match_scores:
                        print(f"      - {name}: match_val={score:.4f} (overlap={overlap:.3f}, n={n:.3f}, g={n_sgwv})")

                # spgrep fallback
                if not matched_labels and HAS_SPGREP:
                    # Create a mini-bcs-table for label matching
                    mini_bcs_table = label_context.adjusted_characters
                    best_spgrep_label = self._label_block_with_spgrep(sg, little_group_indices, block_mats, mini_bcs_table)
                    if best_spgrep_label:
                        matched_labels = [best_spgrep_label]
                        total_irrep_dim = block_size

                labels_for_opd = matched_labels
                if (
                    self._compute_heuristic_opds
                    and not labels_for_opd
                    and heuristic_label
                    and "/" not in heuristic_label
                    and "+" not in heuristic_label
                ):
                    labels_for_opd = [heuristic_label]
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
                else:
                    match_label = heuristic_label

                # Identify OPDs for exact labels and for unambiguous heuristic labels.
                # Ambiguous fallbacks such as P1/P2 are intentionally excluded because
                # they do not define a unique reference irrep basis.
                if labels_for_opd and total_irrep_dim == block_size:
                    try:
                        # Use spgrep for reference matrices.
                        # Both methods now return (ref_mats, spgrep_lg_indices).
                        if len(labels_for_opd) == 1:
                            ref_mats, sp_lg_idx = self._get_reference_matrices(sg, little_group_indices, labels_for_opd[0])
                        else:
                            ref_mats, sp_lg_idx = self._get_combined_reference_matrices(sg, little_group_indices, labels_for_opd)

                        if ref_mats is not None and sp_lg_idx is not None:
                            row_by_sym = {int(isym): i for i, isym in enumerate(little_group_indices)}
                            try:
                                block_mats_ref = np.array([block_mats[row_by_sym[int(idx)]] for idx in sp_lg_idx])
                            except KeyError:
                                block_mats_ref = block_mats
                            U = self._solve_unitary_mapping(ref_mats, block_mats_ref)

                            # Build little group arrays from spgrep's LG indices so
                            # they are consistent with ref_mats (which is indexed over
                            # the spgrep LG, not the irrep-package LG).
                            from .symmetry_identification import get_isotropy_subgroup
                            parent_lattice = self._primitive.cell
                            little_rots = np.array([sg.symmetries[idx].rotation for idx in sp_lg_idx], dtype=int)
                            little_trans = np.array([sg.symmetries[idx].translation for idx in sp_lg_idx], dtype=float)

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

                                    # Collect OPDs from all maximal isotropy subgroups.
                                    # Prefer real OPDs because the reported phonon-mode
                                    # daughters are real distortion directions; complex
                                    # relative phases are useful fallbacks but can add
                                    # extra non-real order-parameter families.
                                    real_opds = []
                                    complex_opds = []
                                    for subgroup_i, indices in enumerate(enumerator.maximal_isotropy_subgroups):
                                        opds_enum = enumerator.order_parameter_directions[subgroup_i]
                                        for opd_vec in opds_enum:
                                            if np.allclose(np.imag(opd_vec), 0, atol=1e-8):
                                                real_opds.append(np.real(opd_vec).astype(complex))
                                            else:
                                                complex_opds.append(opd_vec)

                                    candidate_results = []
                                    seen_daughters = set()
                                    for opd_vec in real_opds + complex_opds:
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
                                            daughter = f"{sg_sym}(#{sg_num})"
                                        except Exception:
                                            daughter = "-"

                                        if daughter in seen_daughters:
                                            continue
                                        seen_daughters.add(daughter)
                                        candidate_results.append((opd_vec, daughter))

                                        if len(candidate_results) >= block_size:
                                            break

                                    if len(candidate_results) < block_size:
                                        for opd_vec in real_opds + complex_opds:
                                            if len(candidate_results) >= block_size:
                                                break
                                            candidate_results.append((opd_vec, "-"))

                                    # Assign distinct OPD families to modes in the block.
                                    for i, (opd_vec, daughter) in enumerate(candidate_results[:block_size]):
                                        opds[i] = self._column_to_opd_symbolic(opd_vec)
                                        isotropy_sgs[i] = daughter
                                except Exception:
                                    pass
                    except Exception:
                        pass

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

    def _get_spgrep_ops(self, sg) -> tuple[np.ndarray, np.ndarray]:
        """Return (rotations, translations) for use with spgrep.

        The irrep package sometimes returns a doubled set of primitive operations
        for centred settings (e.g. SG 141 yields 32 ops instead of the correct 16).
        spgrep requires an irredundant set of primitive operations and raises
        ``ValueError: Unreachable!`` when given the doubled set.  We fall back to
        spglib-derived operations when this happens.
        """
        rotations = np.array([sym.rotation for sym in sg.symmetries], dtype=int)
        translations = np.array([sym.translation for sym in sg.symmetries], dtype=float)
        return rotations, translations

    def _get_spgrep_ops_fallback(self, sg) -> tuple[np.ndarray, np.ndarray] | None:
        """Try spglib-derived primitive ops as a fallback for spgrep.

        Returns (rotations, translations) from spglib.find_primitive, or None if
        the primitive cell can't be determined.
        """
        try:
            import spglib as _spglib
            cell = (
                self._primitive.cell,
                self._primitive.scaled_positions,
                self._primitive.numbers,
            )
            prim = _spglib.find_primitive(cell, symprec=self._symprec)
            if prim is None:
                return None
            sym = _spglib.get_symmetry(prim, symprec=self._symprec)
            if sym is None:
                return None
            return sym["rotations"], sym["translations"]
        except Exception:
            return None

    def _get_reference_matrices(self, sg, little_group_indices: list[int], label: str):
        """Get reference irrep matrices D(g) from spgrep matching the BCS label.

        Returns (ref_mats, spgrep_lg_indices) or (None, None) on failure.
        ref_mats is shaped (n_spgrep_lg, d, d) and indexed over spgrep's little group.
        spgrep_lg_indices are indices into sg.symmetries for those ops.
        """
        if not HAS_SPGREP:
            return None, None

        rotations, translations = self._get_spgrep_ops(sg)

        spgrep_irreps_list = None
        try:
            result = get_spacegroup_irreps_from_primitive_symmetry(
                rotations, translations, self._qpoint
            )
            spgrep_irreps_list = result[0] if isinstance(result, tuple) else result
        except Exception as e:
            if self._log_level > 1:
                print(f"  spgrep call failed (irrep ops): {e}, trying spglib ops")

        if spgrep_irreps_list is None:
            # Fallback: use spglib-derived primitive ops.  This handles cases where
            # the irrep package returns a doubled operation set (e.g. SG 141 M-point).
            fb = self._get_spgrep_ops_fallback(sg)
            if fb is None:
                return None, None
            rotations_fb, translations_fb = fb
            try:
                result = get_spacegroup_irreps_from_primitive_symmetry(
                    rotations_fb, translations_fb, self._qpoint
                )
                spgrep_irreps_list = result[0] if isinstance(result, tuple) else result
                rotations, translations = rotations_fb, translations_fb
            except Exception as e2:
                if self._log_level > 1:
                    print(f"  spgrep fallback also failed: {e2}")
                return None, None

        # Use spgrep's own little group for both character matching and returned matrices.
        # spgrep may find more little-group ops than the irrep package (e.g. at P in BCT
        # cells: 8 spgrep ops vs 2 irrep ops).  Using the full spgrep LG ensures the
        # character vector has enough entries to distinguish all irreps.
        from spgrep.group import get_little_group
        _, _, spgrep_lg_indices = get_little_group(rotations, translations, self._qpoint, self._symprec)
        n_sp_lg = len(spgrep_lg_indices)

        kpname = self._bcs_kpname or "GM"
        bcs_table = sg.get_irreps_from_table(kpname, self._qpoint)
        target_chars_orig = bcs_table.get(label)
        if target_chars_orig is None:
            return None, None

        # Build the target character vector only over operations represented in
        # the BCS table for this k-point.  Some centered/primitive settings have
        # extra primitive little-group operations; treating them as zero-character
        # BCS operations makes valid matches look like half-overlaps.
        selected = [
            (pos, int(idx))
            for pos, idx in enumerate(spgrep_lg_indices)
            if int(idx) + 1 in target_chars_orig
        ]
        if not selected:
            selected = [(pos, int(idx)) for pos, idx in enumerate(spgrep_lg_indices)]

        selected_positions = [pos for pos, _idx in selected]
        selected_lg_indices = [idx for _pos, idx in selected]
        target_char_vec = np.array(
            [target_chars_orig.get(idx + 1, 0) for idx in selected_lg_indices],
            dtype=complex,
        )

        best_irrep_mats = None
        best_match = 0

        for irrep_mats in spgrep_irreps_list:
            # irrep_mats is already indexed over spgrep LG (shape: n_sp_lg × d × d)
            irrep_mats_selected = irrep_mats[selected_positions]
            chars = np.trace(irrep_mats_selected, axis1=1, axis2=2)
            overlap = np.abs(np.vdot(target_char_vec, chars)) / len(selected_lg_indices)
            chars_conj = np.conj(chars)
            overlap_conj = np.abs(np.vdot(target_char_vec, chars_conj)) / len(selected_lg_indices)
            if self._log_level > 1:
                print(
                    f"  Debug _get_ref_mats {label}: overlap={overlap:.4f} "
                    f"overlap_conj={overlap_conj:.4f} chars[:4]={chars[:4]}"
                )
            if overlap_conj > overlap:
                overlap = overlap_conj
                irrep_mats_selected = np.conj(irrep_mats_selected)
            if overlap > best_match and overlap > 0.5:
                best_match = overlap
                best_irrep_mats = irrep_mats_selected

        if self._log_level > 1 and best_irrep_mats is not None:
            print(f"  Matched spgrep irrep for {label}: overlap={best_match:.4f}")

        return best_irrep_mats, selected_lg_indices

    def _get_combined_reference_matrices(self, sg, little_group_indices, labels):
        """Build combined reference matrices for a list of irrep labels (reducible block).

        Returns (combined_ref_mats, spgrep_lg_indices) or (None, None) on failure.
        All component irreps must share the same spgrep little group.
        """
        all_refs = []
        shared_lg_indices = None
        for label in labels:
            ref, sp_lg = self._get_reference_matrices(sg, little_group_indices, label)
            if ref is None:
                return None, None
            if shared_lg_indices is None:
                shared_lg_indices = sp_lg
            all_refs.append(ref)

        # all_refs is list of (n_sp_lg, d_i, d_i); stack into block-diagonal form
        n_lg = all_refs[0].shape[0]
        total_dim = sum(r.shape[1] for r in all_refs)
        combined = np.zeros((n_lg, total_dim, total_dim), dtype=complex)
        curr = 0
        for r in all_refs:
            d = r.shape[1]
            combined[:, curr:curr+d, curr:curr+d] = r
            curr += d
        return combined, shared_lg_indices


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
        # Use the BCS little group: R_bcs @ q_bcs = q_bcs (mod G_bcs).
        # This is larger than the primitive little group when the primitive
        # reciprocal lattice has fewer vectors (e.g. body-centered BCT cells),
        # and it matches the group over which the BCS character table is defined.
        little_group_indices = []
        for i_sym, sym in enumerate(sg.symmetries):
            rot_bcs_lg = np.round(refUCinv @ sym.rotation @ refUC).astype(int)
            diff_bcs = rot_bcs_lg @ q_work - q_work
            if np.allclose(diff_bcs - np.round(diff_bcs), 0, atol=self._symprec):
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
            
            # Transform operation to BCS frame for table matching.
            # Always use BCS frame regardless of whether this is Gamma or not,
            # because IrrepTable.symmetries are always in the BCS conventional cell.
            rot_bcs = np.round(refUCinv @ rot_prim @ refUC).astype(int)
            trans_bcs = refUCinv @ (trans_prim + rot_prim @ shiftUC - shiftUC)
            trans_bcs = np.round(trans_bcs, 10) % 1.0
            
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
                        # Polarization vectors are in Cartesian coordinates; no frame transform needed
                        w_work[k, :, d] = evs_reshaped[k, :, d] * phase_w
            else:
                # Keep 'R'-gauge (v)
                w_work = np.zeros((num_atoms, 3, dim), dtype=complex)
                evs_reshaped = evs.reshape(num_atoms, 3, dim)
                for k in range(num_atoms):
                    for d in range(dim):
                        # Polarization vectors are in Cartesian coordinates; no frame transform needed
                        w_work[k, :, d] = evs_reshaped[k, :, d]

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
