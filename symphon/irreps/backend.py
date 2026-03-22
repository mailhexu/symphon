import numpy as np
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

        matched_labels, total_irrep_dim = self._match_block_to_table(
            kpname, irreps_in_table, block_size, chars_calc, lg_bcs_info
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
                        ref_mats, sp_lg_idx = self._get_reference_matrices(sg, little_group_indices, matched_labels[0])
                    else:
                        ref_mats, sp_lg_idx = self._get_combined_reference_matrices(sg, little_group_indices, matched_labels)

                    if ref_mats is not None and sp_lg_idx is not None:
                        U = self._solve_unitary_mapping(ref_mats, block_mats)
                        parent_lattice = self._primitive.cell
                        little_rots = np.array([sg.symmetries[idx].rotation for idx in sp_lg_idx], dtype=int)
                        little_trans = np.array([sg.symmetries[idx].translation for idx in sp_lg_idx], dtype=float)
                        irrep_dim = ref_mats.shape[1]

                        if U is not None and irrep_dim == 1:
                            unit_vec = np.array([1.0], dtype=complex)
                            opds[0] = self._column_to_opd_symbolic(unit_vec)
                            try:
                                sg_num, sg_sym = get_isotropy_subgroup(
                                    parent_lattice, little_rots, little_trans,
                                    self._qpoint, ref_mats, unit_vec, symprec=self._symprec,
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
                                    small_rep=ref_mats.astype(complex),
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
                                            self._qpoint, ref_mats, opd_vec, symprec=self._symprec,
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

    def _match_block_to_table(self, kpname, irreps_in_table, block_size, chars_calc, lg_bcs_info):
        """Apply the GOT formula to match one block to BCS irreps.

        Returns (matched_labels, total_irrep_dim).
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

            if self._log_level > 1 and irr.kpname == kpname:
                print(f"      - {irr.name}: match_val={np.abs(n):.4f} (overlap={overlap:.3f}, g={n_sgwv})")

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

        for conv in test_conventions:
            self._phase_convention = conv
            if self._log_level > 1:
                print(f"\nTesting phase convention: '{conv}'")

            block_matrices, little_group_indices, _ = self._calculate_phonon_representations(sg, table=table)
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

    def _get_reference_matrices(self, sg, little_group_indices: list[int], label: str):
        """Get reference irrep matrices D(g) from spgrep matching the BCS label.

        Returns (ref_mats, spgrep_lg_indices) or (None, None) on failure.
        ref_mats is shaped (n_spgrep_lg, d, d) and indexed over spgrep's little group.
        spgrep_lg_indices are indices into sg.symmetries for those ops.
        """
        if not HAS_SPGREP:
            return None, None

        rotations = np.array([sym.rotation for sym in sg.symmetries], dtype=int)
        translations = np.array([sym.translation for sym in sg.symmetries], dtype=float)

        try:
            result = get_spacegroup_irreps_from_primitive_symmetry(
                rotations, translations, self._qpoint
            )
            spgrep_irreps_list = result[0] if isinstance(result, tuple) else result
        except Exception as e:
            if self._log_level > 1:
                print(f"  spgrep call failed: {e}")
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

        # Build target character vector over spgrep LG ops.
        # sg.symmetries[idx] == BCS op (idx+1).
        target_char_vec = np.array(
            [target_chars_orig.get(int(idx) + 1, 0) for idx in spgrep_lg_indices],
            dtype=complex,
        )

        best_irrep_mats = None
        best_match = 0

        for irrep_mats in spgrep_irreps_list:
            # irrep_mats is already indexed over spgrep LG (shape: n_sp_lg × d × d)
            chars = np.trace(irrep_mats, axis1=1, axis2=2)
            overlap = np.abs(np.vdot(target_char_vec, chars)) / n_sp_lg
            if self._log_level > 1:
                print(f"  Debug _get_ref_mats {label}: overlap={overlap:.4f} chars[:4]={chars[:4]}")
            if overlap > best_match and overlap > 0.5:
                best_match = overlap
                best_irrep_mats = irrep_mats

        if self._log_level > 1 and best_irrep_mats is not None:
            print(f"  Matched spgrep irrep for {label}: overlap={best_match:.4f}")

        return best_irrep_mats, list(spgrep_lg_indices)

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
