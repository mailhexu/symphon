import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from phonopy.phonon.degeneracy import degenerate_sets
from ase.data import atomic_masses
import logging
import sys
import os
from contextlib import contextmanager
import spglib
from irrep.spacegroup_irreps import SpaceGroupIrreps
from spgrep import get_spacegroup_irreps_from_primitive_symmetry

from ..chiral import opd_to_symbolic
from .projection import (
    solve_unitary_mapping,
    prepare_little_group_ops,
    calculate_phonon_representation_matrices,
    match_block_to_table,
    get_reference_matrices,
    get_combined_reference_matrices
)
HAS_SPGREP = True


@contextmanager
def _suppress_spglib_warnings():
    """Suppress spglib C library warnings printed to stderr."""
    devnull = open(os.devnull, 'w')
    old_stderr = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = old_stderr
        devnull.close()

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
        sg = self._setup_spacegroup()
        table = self._load_irrep_table(sg)
        kpname = self._resolve_bcs_kpname(sg, table, kpname)
        bcs_table = self._fetch_bcs_table(sg, kpname)
        only_multidim, min_irrep_dim = self._detect_only_multidim(bcs_table)
        self._build_degenerate_sets(only_multidim, min_irrep_dim)
        best_irreps, best_conv, best_count = self._find_best_convention(sg, table, kpname)
        self._irreps = best_irreps
        self._phase_convention = best_conv
        if self._log_level > 0:
            print(f"Final phase convention: '{best_conv}' ({best_count}/{len(self._degenerate_sets)} blocks labeled)")
        return True

    def _setup_spacegroup(self):
        """Initialise SpaceGroupIrreps and cache spacegroup metadata."""
        cell = (self._primitive.cell, self._primitive.scaled_positions, self._primitive.numbers)
        sg = SpaceGroupIrreps.from_cell(
            cell=cell,
            spinor=False,
            include_TR=False,
            search_cell=True,
            symprec=self._symprec,
            verbosity=self._log_level,
        )
        self._sg_obj = sg
        from phonopy.structure.symmetry import Symmetry
        self._sg_phonopy = Symmetry(self._primitive, symprec=self._symprec).dataset
        self._spacegroup_symbol = sg.name
        self._spacegroup_number = int(sg.number_str.split('.')[0])
        rotations = np.array([sym.rotation for sym in sg.symmetries], dtype=int)
        with _suppress_spglib_warnings():
            pg_result = spglib.get_pointgroup(rotations)
        self._pointgroup_symbol = pg_result[0] if pg_result else None
        return sg

    def _load_irrep_table(self, sg):
        """Load the BCS IrrepTable for the current space group."""
        try:
            from irreptables.irreps import IrrepTable
        except ImportError:
            from irreptables import IrrepTable
        return IrrepTable(str(self._spacegroup_number), sg.spinor)

    def _resolve_bcs_kpname(self, sg, table, kpname):
        """Find and return the BCS k-point label for self._qpoint."""
        refUC = sg.refUC
        refUCTinv = np.linalg.inv(refUC.T)
        q_bcs = refUC.T @ self._qpoint
        q_bcs = np.where(np.abs(q_bcs) < 1e-8, 0.0, q_bcs)
        self._qpoint_bcs = q_bcs

        found_kpname = None
        for irr in table.irreps:
            k_prim_table = refUCTinv @ irr.k
            diff = k_prim_table - self._qpoint
            if np.allclose(diff - np.round(diff), 0, atol=1e-4):
                found_kpname = irr.kpname
                break

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
            self._bcs_kpname = found_kpname
            return found_kpname
        if kpname is not None:
            return kpname
        if (np.abs(self._qpoint) < self._symprec).all():
            return "GM"
        raise ValueError(f"Could not identify BCS label for q-point {self._qpoint}. Please provide kpname.")

    def _fetch_bcs_table(self, sg, kpname):
        """Fetch the BCS character table for kpname at self._qpoint."""
        try:
            return sg.get_irreps_from_table(kpname, self._qpoint)
        except Exception as e:
            if self._log_level > 0:
                print(f"Error fetching BCS table for {kpname} at {self._qpoint}: {e}")
            raise

    def _detect_only_multidim(self, bcs_table):
        """Return (only_multidim, min_irrep_dim) from the BCS character table."""
        min_irrep_dim = float('inf')
        for label, table_chars in bcs_table.items():
            identity_char = table_chars[1] if 1 in table_chars else list(table_chars.values())[0]
            dim = int(round(abs(identity_char)))
            min_irrep_dim = min(min_irrep_dim, dim)
        only_multidim = (min_irrep_dim > 1)
        if only_multidim and self._log_level > 0:
            print(f"\nDetected k-point with ONLY {min_irrep_dim}D irreps (no 1D irreps)")
            print(f"Force-pairing consecutive modes into {min_irrep_dim}D blocks...")
        if self._log_level > 0:
            print(f"Using phase convention: '{self._phase_convention}'-gauge")
        return only_multidim, min_irrep_dim

    def _build_degenerate_sets(self, only_multidim, min_irrep_dim):
        """Initialise self._degenerate_sets, applying force-pairing when needed."""
        self._degenerate_sets = degenerate_sets(self._freqs)
        if not (only_multidim and min_irrep_dim == 2):
            return
        original_deg_sets = self._degenerate_sets
        max_natural_block = max(len(b) for b in original_deg_sets)
        if max_natural_block < min_irrep_dim:
            # All blocks are singletons – force-pair consecutive modes
            forced_pairs = []
            i = 0
            while i < len(self._freqs):
                if i + 1 < len(self._freqs):
                    forced_pairs.append(tuple([i, i + 1]))
                    i += 2
                else:
                    forced_pairs.append(tuple([i]))
                    i += 1
            self._degenerate_sets = forced_pairs
            if self._log_level > 0:
                singles = sum(1 for b in forced_pairs if len(b) == 1)
                pairs = sum(1 for b in forced_pairs if len(b) == 2)
                print(f"  Original degeneracy: {len(original_deg_sets)} blocks")
                print(f"  Forced pairing: {len(forced_pairs)} blocks")
                print(f"    Singles: {singles}, Pairs: {pairs}")
        else:
            if self._log_level > 0:
                print(f"  Original degeneracy: {len(original_deg_sets)} blocks (size>={max_natural_block}; no force-pairing needed)")

    def _build_lg_bcs_info(self, sg, table, little_group_indices):
        """Build per-little-group-op (bcs_idx, phase_corr) lookup list."""
        refUC = sg.refUC
        shiftUC = sg.shiftUC
        refUCinv = np.linalg.inv(refUC)
        q_bcs = refUC.T @ self._qpoint

        rot_to_bcs = {}
        for i_tab, sym_tab in enumerate(table.symmetries):
            key = tuple(sym_tab.R.flatten().tolist())
            rot_to_bcs[key] = (i_tab + 1, sym_tab.t % 1.0)

        lg_bcs_info = []
        for idx, isym in enumerate(little_group_indices):
            sym = sg.symmetries[isym]
            rot_bcs = np.round(refUCinv @ sym.rotation @ refUC).astype(int)
            trans_bcs_sg = np.round(refUCinv @ (sym.translation + sym.rotation @ shiftUC - shiftUC), 10) % 1.0
            rot_key = tuple(rot_bcs.flatten().tolist())
            if rot_key in rot_to_bcs:
                bcs_idx, t_tab = rot_to_bcs[rot_key]
                dT = trans_bcs_sg - t_tab
                dT = dT - np.round(dT)
                phase_corr = np.exp(2j * np.pi * np.dot(q_bcs, dT))
            else:
                bcs_idx = -1
                phase_corr = 1.0
                if self._log_level > 1:
                    print(f"  GOT: op isym={isym} rot_bcs={rot_bcs.tolist()} NOT IN rot_to_bcs!")
            lg_bcs_info.append((bcs_idx, phase_corr))

        if self._log_level > 1:
            n_unmapped = sum(1 for b, _ in lg_bcs_info if b == -1)
            kpname = self._bcs_kpname
            print(f"  GOT: kpname={kpname}, q_bcs={q_bcs}, LG size={len(little_group_indices)}, unmapped={n_unmapped}")

        return lg_bcs_info

    def _label_block(self, sg, kpname, irreps_in_table, block_idx, block,
                     block_mats, little_group_indices, lg_bcs_info):
        """Label one degenerate block and compute OPD/isotropy data.

        Returns a list of per-mode result dicts and whether the block was labeled.
        """
        from collections import Counter
        from ..symmetry_identification import get_isotropy_subgroup

        block_size = len(block)
        num_little = len(little_group_indices)
        chars_calc = [np.trace(block_mats[i]) for i in range(num_little)]

        if self._log_level > 1:
            print(f"    Block {block_idx} (dim {block_size}): matching with table...")

        matched_labels, total_irrep_dim = match_block_to_table(
            irreps_in_table, block_size, chars_calc, lg_bcs_info
        )

        # spgrep fallback
        if not matched_labels and HAS_SPGREP:
            mini_bcs_table = {irr.name: irr.characters for irr in irreps_in_table}
            best_spgrep_label = self._label_block_with_spgrep(sg, little_group_indices, block_mats, mini_bcs_table)
            if best_spgrep_label:
                matched_labels = [best_spgrep_label]
                total_irrep_dim = block_size

        opds = [None] * block_size
        isotropy_sgs = ["-"] * block_size
        U = None

        if matched_labels:
            labeled = True
            counts = Counter(matched_labels)
            sorted_unique = sorted(counts.keys())
            match_label = "+".join([f"{counts[l]}*{l}" if counts[l] > 1 else l for l in sorted_unique])

            if total_irrep_dim == block_size:
                try:
                    if len(matched_labels) == 1:
                        ref_mats, sp_lg_idx = get_reference_matrices(
                            sg, matched_labels[0], self._qpoint, self._bcs_kpname, self._symprec, self._log_level
                        )
                    else:
                        ref_mats, sp_lg_idx = get_combined_reference_matrices(
                            sg, matched_labels, self._qpoint, self._bcs_kpname, self._symprec, self._log_level
                        )

                    if ref_mats is not None and sp_lg_idx is not None:
                        # Reorder needed only when fallback path remapped indices into
                        # spgrep's ordering; original path already matches caller order.
                        if sp_lg_idx == little_group_indices:
                            reordered_ref = ref_mats
                        else:
                            reorder = [sp_lg_idx.index(idx) for idx in little_group_indices]
                            reordered_ref = ref_mats[reorder]
                        parent_lattice = self._primitive.cell

                        U = solve_unitary_mapping(reordered_ref, block_mats)
                        little_rots = np.array([sg.symmetries[idx].rotation for idx in little_group_indices], dtype=int)
                        little_trans = np.array([sg.symmetries[idx].translation for idx in little_group_indices], dtype=float)
                        irrep_dim = ref_mats.shape[1]

                        if U is not None and irrep_dim == 1:
                            unit_vec = np.array([1.0], dtype=complex)
                            opds[0] = self._column_to_opd_symbolic(unit_vec)
                            try:
                                sg_num, sg_sym = get_isotropy_subgroup(
                                    parent_lattice, little_rots, little_trans,
                                    self._qpoint, reordered_ref, unit_vec, symprec=self._symprec,
                                )
                                isotropy_sgs[0] = f"{sg_sym}(#{sg_num})"
                            except Exception:
                                pass
                        else:
                            try:
                                from spgrep_modulation.isotropy import IsotropyEnumerator
                                enumerator = IsotropyEnumerator(
                                    little_rotations=little_rots,
                                    little_translations=little_trans,
                                    qpoint=self._qpoint,
                                    small_rep=reordered_ref.astype(complex),
                                )
                                all_opds = [
                                    opd_vec
                                    for subgroup_i, _ in enumerate(enumerator.maximal_isotropy_subgroups)
                                    for opd_vec in enumerator.order_parameter_directions[subgroup_i]
                                ]
                                for i in range(min(block_size, len(all_opds))):
                                    opd_vec = all_opds[i]
                                    opds[i] = self._column_to_opd_symbolic(opd_vec)
                                    try:
                                        sg_num, sg_sym = get_isotropy_subgroup(
                                            parent_lattice, little_rots, little_trans,
                                            self._qpoint, reordered_ref, opd_vec, symprec=self._symprec,
                                        )
                                        isotropy_sgs[i] = f"{sg_sym}(#{sg_num})"
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                except Exception:
                    pass
        else:
            labeled = False
            match_label = None

        # Acoustic modes at Gamma have no symmetry breaking
        if np.allclose(self._qpoint, 0.0, atol=1e-5) and block and all(abs(self._freqs[j]) < 0.1 for j in block):
            parent_sg_str = f"{self._spacegroup_symbol}(#{self._spacegroup_number})"
            isotropy_sgs = [parent_sg_str] * block_size

        items = [
            {
                "label": match_label,
                "opd": opds[i] if opds[i] is not None else "-",
                "opd_num": U[:, i] if U is not None else None,
                "daughter_sg": isotropy_sgs[i],
            }
            for i in range(block_size)
        ]
        return items, labeled

        return matched_labels, total_irrep_dim

    def _find_best_convention(self, sg, table, kpname):
        """Try both phase conventions and return the one that labels most blocks.

        Returns (best_irreps, best_conv, best_count).
        """
        best_irreps = None
        best_count = -1
        best_conv = self._phase_convention

        test_conventions = ['r', 'R']
        if self._phase_convention in test_conventions:
            test_conventions.remove(self._phase_convention)
            test_conventions.insert(0, self._phase_convention)

        irreps_in_table = [irr for irr in table.irreps if irr.kpname == kpname]
        
        is_gamma = (np.abs(self._qpoint) < self._symprec).all()
        use_bcs_frame = not is_gamma

        for conv in test_conventions:
            self._phase_convention = conv
            if self._log_level > 1:
                print(f"\nTesting phase convention: '{conv}'")

            # 1. Prepare little group ops
            sym_ops_processed, little_group_indices, mapping_to_table = prepare_little_group_ops(
                primitive=self._primitive,
                qpoint=self._qpoint,
                sg_symmetries=sg.symmetries,
                sg_metadata=self._sg_obj,
                table_symmetries=table.symmetries,
                symprec=self._symprec,
                log_level=self._log_level
            )

            # 2. Calculate representation matrices
            block_matrices = calculate_phonon_representation_matrices(
                primitive=self._primitive,
                qpoint=self._qpoint,
                little_group_ops=sym_ops_processed,
                degenerate_sets=self._degenerate_sets,
                eigvecs=self._eigvecs,
                phase_convention=self._phase_convention,
                use_bcs_frame=use_bcs_frame,
                sg_metadata=self._sg_obj
            )
            
            lg_bcs_info = self._build_lg_bcs_info(sg, table, little_group_indices)

            current_irreps = []
            labeled_count = 0
            for block_idx, block in enumerate(self._degenerate_sets):
                items, labeled = self._label_block(
                    sg, kpname, irreps_in_table, block_idx, block,
                    block_matrices[block_idx], little_group_indices, lg_bcs_info,
                )
                current_irreps.extend(items)
                if labeled:
                    labeled_count += 1

            if labeled_count > best_count:
                best_count = labeled_count
                best_irreps = current_irreps
                best_conv = conv

            if labeled_count == len(self._degenerate_sets):
                break

        return best_irreps, best_conv, best_count


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
