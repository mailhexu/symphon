import numpy as np
from typing import Optional, List, cast
from scipy.constants import c, h, e, tera
from phonopy.phonon.irreps import IrReps, IrRepLabels
from phonopy.structure.symmetry import Symmetry
from phonopy.phonon.character_table import character_table
from symphon.abipy_io import read_phbst_freqs_and_eigvecs, ase_atoms_to_phonopy_atoms
from phonopy.phonon.degeneracy import degenerate_sets as get_degenerate_sets
from phonopy.structure.cells import is_primitive_cell
from phonopy import load as phonopy_load
from .chiral_transitions import ChiralTransitionFinder, is_sohncke, get_sohncke_class, SohnckeClass


class ReportingMixin:
    """Mixin class for consistent reporting output."""

    def get_summary_table(self):
        """Return core mode information as list of dicts.

        Each entry contains:
        - qpoint: 3-tuple of fractional coordinates
        - band_index: mode index (0-based)
        - frequency_thz: frequency in THz
        - frequency_cm1: frequency in cm^-1
        - label: irrep label if available (otherwise None)
        - is_ir_active: bool
        - is_raman_active: bool

        ``run()`` must be called before using this method.
        """
        if not hasattr(self, "_freqs"):
            raise RuntimeError("run() must be called before get_summary_table().")

        q = tuple(float(x) for x in self._qpoint)

        freqs_thz = self._freqs
        conv = tera / (c * 100)  # THz -> cm^-1
        n_modes = len(freqs_thz)

        irreps = getattr(self, "_irreps", None)

        # Build IR/Raman activity maps from _RamanIR_labels when available.
        ir_active_map: dict[str, bool] = {}
        raman_active_map: dict[str, bool] = {}

        raman_ir = getattr(self, "_RamanIR_labels", None)
        if raman_ir is not None:
            ir_labels, raman_labels = raman_ir
            for lbl in ir_labels:
                ir_active_map[lbl] = True
            for lbl in raman_labels:
                raman_active_map[lbl] = True

        # Extract labels and OPDs using degenerate sets and _ir_labels when using phonopy backend.
        # If _irreps is already list of dicts (irrep backend), we use it directly.
        raw_labels = cast(List[Optional[str]], [None] * n_modes)
        raw_opds = cast(List[Optional[str]], [None] * n_modes)
        ir_labels_seq = getattr(self, "_ir_labels", None)
        deg_sets = getattr(self, "_degenerate_sets", None)
        
        # New: Chiral transitions mapping
        chiral_map = getattr(self, "_chiral_transitions_map", {})

        # Build a mapping from mode index to degenerate set index
        mode_to_degset: dict[int, int] = {}
        if deg_sets is not None:
            for set_idx, deg_set in enumerate(deg_sets):
                for mode_idx in deg_set:
                    mode_to_degset[mode_idx] = set_idx

        for band_index in range(n_modes):
            label = None
            opd = None

            # 1) Prefer label attached to irreps[band_index] when present.
            if irreps is not None and band_index < len(irreps):
                ir = irreps[band_index]
                if hasattr(ir, "label"):
                    label = ir.label
                elif isinstance(ir, dict) and "label" in ir:
                    label = ir["label"]
                
                if isinstance(ir, dict) and "opd" in ir:
                    opd = ir["opd"]
                elif hasattr(ir, "opd"):
                    opd = ir.opd

            # 2) Fallback: use _ir_labels indexed by degenerate set.
            if label is None and ir_labels_seq is not None:
                # Find which degenerate set this mode belongs to
                set_idx = mode_to_degset.get(band_index)
                if set_idx is not None and set_idx < len(ir_labels_seq):
                    cand = ir_labels_seq[set_idx]
                    # Extract the label string
                    if isinstance(cand, (tuple, list)) and cand:
                        label = cand[0]
                    elif isinstance(cand, str):
                        label = cand

            # 3) Check BCS specific OPD
            if opd is None:
                opd_bcs_seq = getattr(self, "_irrep_opds_bcs", None)
                if opd_bcs_seq and band_index < len(opd_bcs_seq):
                    opd = opd_bcs_seq[band_index]

            raw_labels[band_index] = label
            raw_opds[band_index] = opd

        # Propagate labels within degenerate sets to ensure all members
        # of a multiplet share the same label.
        if deg_sets is not None:
            for deg_set in deg_sets:
                labels_in_set = {raw_labels[i] for i in deg_set if raw_labels[i]}
                if len(labels_in_set) == 1:
                    lbl = labels_in_set.pop()
                    for i in deg_set:
                        raw_labels[i] = lbl
                # If 0 or >1 distinct labels, leave as-is (ambiguous).

        # Third pass: build summary rows using final labels and IR/Raman flags.
        summary = []
        for band_index, f_thz in enumerate(freqs_thz):
            freq_thz = float(f_thz)
            freq_cm1 = freq_thz * conv

            label = raw_labels[band_index]
            is_ir_active = bool(label and ir_active_map.get(label, False))
            is_raman_active = bool(label and raman_active_map.get(label, False))

            # Lookup chiral transitions
            opd_str = raw_opds[band_index] or "-"
            daughter_str = "-"
            # 4) Prefer daughter_sg from backend if present
            backend = getattr(self, "_irrep_backend_obj", None)
            if backend and hasattr(backend, "_irreps") and band_index < len(backend._irreps):
                ir = backend._irreps[band_index]
                if isinstance(ir, dict) and "daughter_sg" in ir:
                    daughter_str = ir["daughter_sg"]
                elif hasattr(ir, "daughter_sg"):
                    daughter_str = ir.daughter_sg
            elif irreps is not None and band_index < len(irreps):
                ir = irreps[band_index]
                if isinstance(ir, dict) and "daughter_sg" in ir:
                    daughter_str = ir["daughter_sg"]
                elif hasattr(ir, "daughter_sg"):
                    daughter_str = ir.daughter_sg

            # 5) Lookup chiral transitions if daughter_str is still "-"
            if daughter_str == "-":
                labels_to_check = []
                if label:
                    labels_to_check.append(label)
                
                irrep_labels_bcs = getattr(self, "_irrep_labels_bcs", None)
                if irrep_labels_bcs and band_index < len(irrep_labels_bcs):
                    label_bcs = irrep_labels_bcs[band_index]
                    if label_bcs and label_bcs not in labels_to_check:
                        labels_to_check.append(label_bcs)

                for lbl in labels_to_check:
                    # Try exact match first
                    trans_list = chiral_map.get(lbl)
                    if not trans_list:
                        # Try match without branch index (e.g. Z3:1 -> Z3)
                        base_label = lbl.split(":")[0]
                        trans_list = chiral_map.get(base_label)
                    
                    if trans_list:
                        if not isinstance(trans_list, list):
                            trans_list = [trans_list]
                        
                        if opd_str == "-":
                            # If backend didn't provide OPD, use symbolic from transitions
                            opds = sorted(list(set(t.opd.symbolic for t in trans_list)))
                            opd_str = ", ".join(opds)
                            
                        daughters = sorted(list(set(f"{t.daughter_spg_symbol}(#{t.daughter_spg_number})" for t in trans_list)))
                        daughter_str = ", ".join(daughters)
                        break # Found a match

            summary.append(
                {
                    "qpoint": q,
                    "band_index": band_index,
                    "frequency_thz": freq_thz,
                    "frequency_cm1": freq_cm1,
                    "label": label,
                    "is_ir_active": is_ir_active,
                    "is_raman_active": is_raman_active,
                    "opd": opd_str,
                    "daughter_sg": daughter_str,
                }
            )

        return summary

    def _compute_chiral_transitions(self):
        """Compute possible chiral transitions for current space group and q-point."""
        spg_number = getattr(self, "_spacegroup_number", None)
        if spg_number is None or is_sohncke(spg_number):
            # Parent is already chiral or unknown
            self._chiral_transitions_map = {}
            return

        # Use find_chiral_transitions which is comprehensive
        try:
            finder = ChiralTransitionFinder(spg_number, symprec=self._symprec)
            
            # We need to provide the q-point coordinates
            q = self._qpoint
            
            # We don't have a reliable BCS label here, so we let it search or use "current"
            # If kpname was provided during run(), we might have it stored.
            kpname = getattr(self, "_bcs_kpname", None)
            transitions = finder.find_chiral_transitions(qpoint=q, qpoint_label=kpname)
            
            # Group transitions by irrep label
            mapping = {}
            for t in transitions:
                lbl = t.irrep_label
                if lbl not in mapping:
                    mapping[lbl] = []
                mapping[lbl].append(t)
            
            self._chiral_transitions_map = mapping
        except Exception as e:
            if getattr(self, "_log_level", 0) > 0:
                print(f"Warning: Failed to compute chiral transitions: {e}")
            self._chiral_transitions_map = {}

    def format_summary_table(self, include_header: bool = True, include_symmetry: bool = True, include_qpoint_cols: bool = True, show_chiral: bool = False) -> str:
        """Format the summary table as a human-readable string.
        
        Args:
            include_header: Whether to include column headers
            include_symmetry: Whether to include q-point, space group, and point group info
            include_qpoint_cols: Whether to include qx, qy, qz columns in the table
            show_chiral: Whether to include OPD and daughter SG chiral transition columns
        """
        summary = self.get_summary_table()
        show_chiral_cols = any(row.get("daughter_sg") != "-" for row in summary)
        
        # Decide which columns to show.
        # OPD column should show if show_chiral is requested OR if BCS labels/OPDs were computed.
        bcs_labels = getattr(self, "_irrep_labels_bcs", [])
        bcs_opds = getattr(self, "_irrep_opds_bcs", [])
        has_bcs_data = any(lbl for lbl in bcs_labels) or any(opd for opd in bcs_opds)
        
        show_both = has_bcs_data
        show_activity = True

        lines = []
        if include_symmetry:
            if summary:
                qx, qy, qz = summary[0]["qpoint"]
                lines.append(f"q-point: [{qx:.4f}, {qy:.4f}, {qz:.4f}]")
            
            space_group = getattr(self, "_spacegroup_symbol", None)
            point_group = getattr(self, "_pointgroup_symbol", None)
            
            if space_group:
                lines.append(f"Space group: {space_group}")
            if point_group:
                lines.append(f"Point group: {point_group}")
            if lines:
                lines.append("")
        if include_header:
            if include_qpoint_cols:
                header = f"{'# qx':>8s} {'qy':>8s} {'qz':>8s} {'band':>5s} {'freq(THz)':>10s} {'freq(cm-1)':>11s}"
                if show_both:
                    header += f" {'label(M)':>12s} {'label(BCS)':>12s}"
                else:
                    header += f" {'label':>12s}"
                if show_activity:
                    header += f" {'IR':^4s} {'Raman':^6s}"
            else:
                header = f"{'# band':>6s} {'freq(THz)':>10s} {'freq(cm-1)':>11s}"
                if show_both:
                    header += f" {'label(M)':>12s} {'label(BCS)':>12s}"
                else:
                    header += f" {'label':>12s}"
                if show_activity:
                    header += f" {'IR':^4s} {'Raman':^6s}"
            
            if show_chiral_cols:
                header += f" {'OPD':<15s} {'Daughter SG':<20s} {'Chiral':<8s}"
            elif show_both:
                header += f" {'OPD':<15s}"
            lines.append(header)

        for i, row in enumerate(summary):
            qx, qy, qz = row["qpoint"]
            bi = row["band_index"]
            f_thz = row["frequency_thz"]
            f_cm1 = row["frequency_cm1"]
            label = row["label"] or "-"
            ir_flag = "Y" if row.get("is_ir_active") else "."
            raman_flag = "Y" if row.get("is_raman_active") else "."
            
            if include_qpoint_cols:
                line = f"{qx:8.4f} {qy:8.4f} {qz:8.4f} {bi:5d} {f_thz:10.4f} {f_cm1:11.2f}"
            else:
                line = f"{bi:6d} {f_thz:10.4f} {f_cm1:11.2f}"

            if show_both:
                irrep_labels_bcs = getattr(self, "_irrep_labels_bcs", None)
                label_mulliken = label if label else "-"
                label_bcs = "-"
                if irrep_labels_bcs and i < len(irrep_labels_bcs):
                    label_bcs = irrep_labels_bcs[i] or "-"
                line += f" {str(label_mulliken):>12s} {str(label_bcs):>12s}"
            else:
                line += f" {str(label):>12s}"
                
            if show_activity:
                line += f" {ir_flag:^4s} {raman_flag:^6s}"
            
            if show_chiral_cols:
                opd = row.get("opd") or "-"
                dsg = row.get("daughter_sg") or "-"
                line += f" {str(opd):<15s} {str(dsg):<20s}"
                
                # Check if daughter SG is Class II (chiral with pair)
                if dsg != "-" and "(#" in dsg:
                    try:
                        # Extract SG number from string like "P2_12_12_1 (#19)" or "P4mm(#99)"
                        import re
                        match = re.search(r"\(#(\d+)\)", dsg)
                        if match:
                            sg_num = int(match.group(1))
                            sohncke_cls = get_sohncke_class(sg_num)
                            if sohncke_cls == SohnckeClass.CLASS_II:
                                line += f" {'II-pair':<8s}"
                            elif sohncke_cls == SohnckeClass.CLASS_III:
                                line += f" {'III':<8s}"
                            else:
                                line += f" {'-':<8s}"
                        else:
                            line += f" {'-':<8s}"
                    except (ValueError, IndexError):
                        line += f" {'-':<8s}"
                else:
                    line += f" {'-':<8s}"
            elif show_both:
                opd = row.get("opd") or "-"
                line += f" {str(opd):<15s}"
                
            lines.append(line)

        return "\n".join(lines)

    def get_verbose_output(self) -> str:
        """Get verbose phonopy-style output."""

        from io import StringIO
        import contextlib
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            show_method = getattr(self, "_show", None) or getattr(self, "show", None)
            if show_method is None:
                print(repr(self))
            else:
                try:
                    show_method(True)
                except TypeError:
                    show_method()
        return buf.getvalue()


class IrRepsEigen(IrReps, IrRepLabels, ReportingMixin):
    def __init__(
        self,
        primitive_atoms,
        qpoint,
        freqs,
        eigvecs,
        symprec: float = 1e-5,
        degeneracy_tolerance: float = 1e-5,
        log_level: int = 0,
    ) -> None:
        self._is_little_cogroup = True  # Always use little_cogroup
        self._log_level = log_level

        self._qpoint = np.array(qpoint)
        self._degeneracy_tolerance = degeneracy_tolerance
        self._symprec = symprec
        self._irrep_opds_num_bcs = []
        self._primitive = primitive_atoms
        self._freqs, self._eig_vecs = freqs, eigvecs
        self._character_table = None
        self._verbose = False
        self._irrep_labels_bcs: List[Optional[str]] = []
        self._irrep_opds_bcs: List[Optional[str]] = []
        self._irrep_backend_obj = None
        self._chiral_transitions_map = {}
        self._spacegroup_number = None
        self._compute_chiral = False

    def run(self, kpname=None) -> bool:
        self._bcs_kpname = kpname
        # 1. Phonopy logic (for Mulliken labels, IR/Raman, and general setup)
        self._symmetry_dataset = Symmetry(self._primitive, symprec=self._symprec).dataset
        if not is_primitive_cell(self._symmetry_dataset.rotations):
            raise RuntimeError(
                "Non-primitve cell is used. Your unit cell may be transformed to "
                "a primitive cell by PRIMITIVE_AXIS tag."
            )

        (self._rotations_at_q, self._translations_at_q) = self._get_rotations_at_q()

        self._g = len(self._rotations_at_q)

        import spglib
        self._pointgroup_symbol, _, _ = spglib.get_pointgroup(self._rotations_at_q)
        
        # Get space group symbol from symmetry dataset
        self._spacegroup_symbol = self._symmetry_dataset.international
        self._spacegroup_number = self._symmetry_dataset.number

        (self._transformation_matrix, self._conventional_rotations,) = self._get_conventional_rotations()

        self._ground_matrices = self._get_ground_matrix()
        self._degenerate_sets = self._get_degenerate_sets()
        self._irreps = self._get_irreps()
        self._characters, self._irrep_dims = self._get_characters()

        self._ir_labels = None
        self._RamanIR_labels = None

        if (
            self._pointgroup_symbol in character_table.keys()
            and character_table[self._pointgroup_symbol] is not None
        ):
            try:
                self._rotation_symbols, character_table_of_ptg = self._get_rotation_symbols(self._pointgroup_symbol)
                self._character_table = character_table_of_ptg
                if self._rotation_symbols:
                    self._ir_labels = self._get_irrep_labels(character_table_of_ptg)
                    if (abs(self._qpoint) < self._symprec).all():
                        self._RamanIR_labels = self._get_infrared_raman()
                        IR_labels, Ram_labels = self._RamanIR_labels
                        if self._log_level > 0:
                            print("IR  labels", IR_labels)
                            print("Ram labels", Ram_labels)
                elif (abs(self._qpoint) < self._symprec).all():
                    if self._log_level > 0:
                        print("Database for this point group is not preprared.")
                else:
                    if self._log_level > 0:
                        print(f"Database for point group {self._pointgroup_symbol} at non-Gamma point is not prepared.")
            except Exception as e:
                self._rotation_symbols = None
                if self._log_level > 0:
                    print(f"Could not determine rotation symbols for {self._pointgroup_symbol} at {self._qpoint}: {e}")
        else:
            self._rotation_symbols = None
            if self._log_level > 0:
                print(f"Point group {self._pointgroup_symbol} not found in database.")

        # 2. Always run irrep backend for BCS notation
        from .irrep_backend import IrRepsIrrep
        self._irrep_backend_obj = IrRepsIrrep(
            primitive=self._primitive,
            qpoint=self._qpoint,
            freqs=self._freqs,
            eigvecs=self._eig_vecs,
            symprec=self._symprec,
            log_level=self._log_level
        )
        self._irrep_backend_obj.run(kpname=kpname)
        self._irrep_labels_bcs = []
        self._irrep_opds_bcs = []
        self._irrep_opds_num_bcs = []
        for irrep in self._irrep_backend_obj._irreps:
            if isinstance(irrep, dict):
                self._irrep_labels_bcs.append(irrep.get("label"))
                self._irrep_opds_bcs.append(irrep.get("opd"))
                self._irrep_opds_num_bcs.append(irrep.get("opd_num"))
            else:
                self._irrep_labels_bcs.append(getattr(irrep, "label", None))
                self._irrep_opds_bcs.append(getattr(irrep, "opd", None))
                self._irrep_opds_num_bcs.append(getattr(irrep, "opd_num", None))

        if getattr(self, "_compute_chiral", False):
            self._compute_chiral_transitions()
        else:
            self._chiral_transitions_map = {}
        return True

    def get_modulated_supercell(
        self,
        mode_index: int,
        amplitude: float = 0.1,
        supercell_matrix: Optional[np.ndarray] = None,
    ):
        """
        Generate a modulated supercell for a specific phonon mode.
        If the mode is part of a degenerate set, it uses the high-symmetry basis
        calculated during the run() phase.
        
        Args:
            mode_index: Index of the phonon mode (0 to 3N-1).
            amplitude: Maximum displacement in Angstrom.
            supercell_matrix: (3, 3) matrix for supercell generation. 
                             If None, a minimal commensurate supercell is used.
                             
        Returns:
            PhonopyAtoms object representing the modulated structure.
        """
        import numpy as np
        from phonopy.structure.cells import get_supercell
        
        if supercell_matrix is None:
            if np.allclose(self._qpoint, 0):
                supercell_matrix = np.eye(3, dtype=int)
            else:
                # Simple estimation of commensurate supercell
                from fractions import Fraction
                denoms = [Fraction(x).limit_denominator(100).denominator for x in self._qpoint]
                supercell_matrix = np.diag(denoms)
        
        supercell_matrix = np.array(supercell_matrix)
        if supercell_matrix.ndim == 1:
            supercell_matrix = np.diag(supercell_matrix)
            
        sc = get_supercell(self._primitive, supercell_matrix)
        sc_size = np.abs(np.around(np.linalg.det(supercell_matrix)))
        
        num_atoms = len(self._primitive.masses)
        # self._eig_vecs is (3*num_atoms, 3*num_atoms)
        e_full = self._eig_vecs[:, mode_index]
        e_reshaped = e_full.reshape(num_atoms, 3)
        
        m = sc.masses
        s2u_map = sc.s2u_map
        spos = sc.scaled_positions
        
        # Calculation follows spgrep-modulation phase convention
        # R_l + r_basis = spos @ supercell_lattice
        # q . (spos @ supercell_lattice) = q . (spos @ supercell_matrix @ primitive_lattice)
        # Since q is in primitive reciprocal basis: q . (x @ primitive_lattice) = q_frac . x_frac
        phases = np.exp(2j * np.pi * np.dot(np.dot(spos, supercell_matrix.T), self._qpoint))
        
        # u_jl = e_j * exp(...) / sqrt(m_j * sc_size)
        disps = e_reshaped[s2u_map] * (phases[:, None] / np.sqrt(m[:, None] * sc_size))
        
        # Real part is the actual displacement
        real_disps = np.real(disps)
        max_d = np.max(np.linalg.norm(real_disps, axis=1))
        if max_d > 1e-10:
            real_disps *= (amplitude / max_d)
            
        # Create a copy and apply displacements
        modulated_sc = sc.copy()
        modulated_sc.positions += real_disps
        
        return modulated_sc


    # --- New Query Methods ---
    
    def get_spacegroup(self) -> str:
        """Get the space group symbol of the parent structure."""
        return getattr(self, "_spacegroup_symbol", None)
        
    def get_frequencies(self, unit: str = "THz"):
        """
        Return the frequencies for the q-point in the specified unit.
        Supported units: "THz", "eV", "meV", "cm-1".
        """
        unit = unit.lower()
        if unit == "thz":
            return self._freqs
        elif unit == "cm-1":
            # THz to cm^-1: 1 THz = 10^12 Hz. Wavenumber = f / c. In cm^-1 it is f / (c * 100)
            return self._freqs * tera / (c * 100)
        elif unit == "ev":
            # THz to eV: E = h * f. In eV it is h * f / e
            return self._freqs * tera * h / e
        elif unit == "mev":
            # THz to meV
            return self._freqs * tera * h / e * 1000
        else:
            raise ValueError(f"Unsupported unit: {unit}. Use 'THz', 'eV', 'meV', or 'cm-1'.")

    def get_eigenvalues(self, unit: str = "THz"):
        """
        Return the eigenvalues (signed omega^2) for the q-point based on the specified unit.
        Supported units for omega: "THz", "eV", "meV", "cm-1".
        """
        freqs = self.get_frequencies(unit=unit)
        return np.sign(freqs) * (freqs ** 2)

    def get_eigenvectors(self):
        """Return the eigenvectors for the q-point."""
        return self._eig_vecs

    def get_eigendisplacements(self):
        """Return the eigendisplacements for the q-point."""
        # Displacements are eigenvectors divided by sqrt(mass)
        masses = self._primitive.masses
        eigvecs = self._eig_vecs  # In phonopy, shape is usually (num_atoms*3, num_bands) or vice versa.
        num_atom = len(masses)
        disps = np.zeros_like(eigvecs)
        
        # Check shape dynamically
        is_transposed = eigvecs.shape[0] == num_atom * 3
        
        for j in range(num_atom):
            mass_factor = np.sqrt(masses[j])
            if is_transposed:
                disps[j*3:(j+1)*3, :] = eigvecs[j*3:(j+1)*3, :] / mass_factor
            else:
                disps[:, j*3:(j+1)*3] = eigvecs[:, j*3:(j+1)*3] / mass_factor
        return disps

    def _get_labels_list(self, label_attr) -> List[Optional[str]]:
        # Helper to unpack degenerate labels
        raw_labels = cast(List[Optional[str]], [None] * len(self._freqs))
        seq = getattr(self, label_attr, None)
        if seq is None:
            return raw_labels
        
        mode_to_degset: dict[int, int] = {}
        if self._degenerate_sets is not None:
            for set_idx, deg_set in enumerate(self._degenerate_sets):
                for mode_idx in deg_set:
                    mode_to_degset[mode_idx] = set_idx
                    
        for band_index in range(len(self._freqs)):
            set_idx = mode_to_degset.get(band_index)
            if set_idx is not None and set_idx < len(seq):
                cand = seq[set_idx]
                if isinstance(cand, (tuple, list)) and cand:
                    raw_labels[band_index] = cand[0]
                elif isinstance(cand, str):
                    raw_labels[band_index] = cand
        return raw_labels

    def get_mulliken_labels(self) -> list[str | None]:
        """Get the Mulliken labels for all modes (only for Gamma point)."""
        return self._get_labels_list("_ir_labels")

    def get_bcs_labels(self) -> list[str | None]:
        """Get the BCS labels for all modes."""
        if not hasattr(self, "_irrep_labels_bcs") or not self._irrep_labels_bcs:
            return [None] * len(self._freqs)
        return self._irrep_labels_bcs

    def get_mulliken_label(self, mode_index: int) -> str | None:
        """Get the Mulliken label for a specific mode (only for Gamma point)."""
        labels = self.get_mulliken_labels()
        return labels[mode_index] if mode_index < len(labels) else None

    def get_bcs_label(self, mode_index: int) -> str | None:
        """Get the BCS label for a specific mode."""
        labels = self.get_bcs_labels()
        return labels[mode_index] if mode_index < len(labels) else None

    def is_ir_active(self, mode_index: int) -> bool:
        """Check if a specific mode is IR active (only for Gamma point)."""
        label = self.get_mulliken_label(mode_index)
        if not label or not hasattr(self, "_RamanIR_labels") or not self._RamanIR_labels:
            return False
        return label in self._RamanIR_labels[0]

    def is_raman_active(self, mode_index: int) -> bool:
        """Check if a specific mode is Raman active (only for Gamma point)."""
        label = self.get_mulliken_label(mode_index)
        if not label or not hasattr(self, "_RamanIR_labels") or not self._RamanIR_labels:
            return False
        return label in self._RamanIR_labels[1]

    def get_ir_indices(self) -> list[int]:
        """Return indices of all IR active modes at Gamma."""
        return [i for i in range(len(self._freqs)) if self.is_ir_active(i)]

    def get_raman_indices(self) -> list[int]:
        """Return indices of all Raman active modes at Gamma."""
        return [i for i in range(len(self._freqs)) if self.is_raman_active(i)]

    def get_indices_by_mulliken(self, label: str) -> list[int]:
        """Return indices of modes with a specific Mulliken label at Gamma."""
        labels = self._get_labels_list("_ir_labels")
        return [i for i, lbl in enumerate(labels) if lbl == label]

    def get_indices_by_bcs(self, label: str) -> tuple[np.ndarray, str | None, list[int]]:
        """Return the q-point coordinates, q-point name (if available), and indices of modes with a specific BCS label."""
        indices = []
        if hasattr(self, "_irrep_labels_bcs") and self._irrep_labels_bcs:
            indices = [i for i, lbl in enumerate(self._irrep_labels_bcs) if lbl == label]
        
        # In a single-qpoint class, the q-point name might not be stored directly if not passed in run,
        # but we can try to return what we have (we usually pass kpname in run()).
        kpname = getattr(self, "_bcs_kpname", None) # Wait, is _bcs_kpname stored? Let's check run.
        # Actually in run() we didn't store kpname in self._bcs_kpname except implicitly in chiral transitions.
        # Let's just retrieve it from _irrep_backend_obj if available.
        if kpname is None and self._irrep_backend_obj and hasattr(self._irrep_backend_obj, "kpname"):
            kpname = self._irrep_backend_obj.kpname
        return (self._qpoint, kpname, indices)

    def _get_degenerate_sets(self):
        deg_sets = get_degenerate_sets(self._freqs, cutoff=self._degeneracy_tolerance)
        return deg_sets

    def _get_infrared_raman(self):
        """Compute IR- and Raman-active irreps using symmetry operations.

        Once irreps and characters are available, use them together with
        symmetry operations to determine which irreps are IR- and
        Raman-active.
        """
        # Multiplicity formula: n_i = 1/g * sum_R chi_i(R)* * chi_reducible(R)
        # For IR activity, chi_reducible(R) = Tr(R_cart)
        # For Raman activity, chi_reducible(R) = 1/2 * [Tr(R_cart)^2 + Tr(R_cart^2)]
        
        # In any basis (including fractional), the trace is invariant.
        # So we can use the character table's mapping_table matrices directly.
        
        ir_active = set()
        raman_active = set()
        
        if self._pointgroup_symbol not in character_table:
            return ir_active, raman_active
            
        # character_table[symbol] is a list of table variants. 
        # Usually we just need the first one that matches our rotations.
        # Phonopy's _get_rotation_symbols already found the correct one and
        # stored it in self._character_table.
        
        if not self._character_table:
            return ir_active, raman_active

        # 1. Precalculate characters of reducible representations for each class
        mapping = self._character_table["mapping_table"]
        g = 0
        chi_ir_class = []
        chi_raman_class = []
        
        for op_class in mapping:
            ops = mapping[op_class]
            g += len(ops)
            # All ops in a class have same trace
            R = np.array(ops[0])
            tr_R = np.trace(R)
            chi_ir_class.append(tr_R)
            chi_raman_class.append(0.5 * (tr_R**2 + np.trace(np.dot(R, R))))
            
        # 2. Identify active irreps
        for label, irrep_chars in self._character_table["character_table"].items():
            n_ir = 0
            n_ram = 0
            for iclass, op_class in enumerate(mapping):
                degen = len(mapping[op_class])
                n_ir += np.conj(irrep_chars[iclass]) * chi_ir_class[iclass] * degen
                n_ram += np.conj(irrep_chars[iclass]) * chi_raman_class[iclass] * degen
            
            n_ir = np.abs(n_ir) / g
            n_ram = np.abs(n_ram) / g
            
            if n_ir > 0.5:
                ir_active.add(label)
            if n_ram > 0.5:
                raman_active.add(label)
                
        return ir_active, raman_active


class IrRepsPhonopy(IrRepsEigen):
    """Irreps helper for direct phonopy calculations."""

    def __init__(
        self,
        phonopy_params,
        qpoint,
        symprec: float | None = None,
        degeneracy_tolerance: float = 1e-5,
        log_level: int = 0,
    ) -> None:
        phonon = phonopy_load(phonopy_params)
        q = np.asarray(qpoint, dtype=float)
        phonon.run_qpoints([q], with_eigenvectors=True)
        q_dict = phonon.get_qpoints_dict()
        freqs = np.array(q_dict["frequencies"][0], dtype=float)
        eigvecs = np.array(q_dict["eigenvectors"][0], dtype=complex)
        primitive_atoms = phonon.primitive

        if symprec is None:
            symprec = phonon._symprec

        super().__init__(
            primitive_atoms,
            qpoint,
            freqs,
            eigvecs,
            symprec=symprec,
            degeneracy_tolerance=degeneracy_tolerance,
            log_level=log_level,
        )


class IrRepsAnaddb(IrRepsEigen):
    """Irreps helper tied to anaddb PHBST output."""

    def __init__(
        self,
        phbst_fname,
        ind_q,
        symprec: float = 1e-5,
        degeneracy_tolerance: float = 1e-5,
        log_level: int = 0,
    ) -> None:
        atoms, qpoints, freqs, eig_vecs = read_phbst_freqs_and_eigvecs(phbst_fname)
        primitive_atoms = ase_atoms_to_phonopy_atoms(atoms)

        super().__init__(
            primitive_atoms,
            qpoints[ind_q],
            freqs[ind_q],
            eig_vecs[ind_q],
            symprec=symprec,
            degeneracy_tolerance=degeneracy_tolerance,
            log_level=log_level,
        )


def print_irreps(
    phbst_fname,
    ind_q,
    symprec=1e-5,
    degeneracy_tolerance=1e-4,
    log_level=0,
    show_verbose=False,
    kpname=None,
):
    irr = IrRepsAnaddb(
        phbst_fname=phbst_fname,
        ind_q=ind_q,
        symprec=symprec,
        degeneracy_tolerance=degeneracy_tolerance,
        log_level=log_level,
    )
    irr.run(kpname=kpname)

    # Print summary table
    print(irr.format_summary_table())

    # Optionally print verbose output
    if show_verbose:
        print()
        print("# Verbose irreps output")
        print(irr.get_verbose_output())

    return irr


def print_irreps_phonopy(
    phonopy_params,
    qpoint,
    symprec: float | None = None,
    degeneracy_tolerance: float = 1e-4,
    log_level: int = 0,
    show_verbose: bool = False,
    kpname=None,
):
    irr = IrRepsPhonopy(
        phonopy_params=phonopy_params,
        qpoint=qpoint,
        symprec=symprec,
        degeneracy_tolerance=degeneracy_tolerance,
        log_level=log_level,
    )
    irr.run(kpname=kpname)

    # Print summary table
    print(irr.format_summary_table())

    # Optionally print verbose output
    if show_verbose:
        print()
        print("# Verbose irreps output")
        print(irr.get_verbose_output())

    return irr


def get_special_qpoints(primitive_atoms, symprec=1e-5) -> list[dict]:
    """
    Get all special q-points for a given primitive structure.
    Uses the 'irrep' package to find the BCS special q-points and transforms
    them to the input structure's reciprocal basis.
    
    Returns:
        List of dictionaries with keys:
        - label: The BCS q-point label (e.g., 'GM', 'X')
        - qpoint_bcs: The q-point coordinates in the standard BCS reciprocal cell
        - qpoint_input: The q-point coordinates in the input cell's reciprocal basis
    """
    try:
        from irreptables.irreps import IrrepTable
    except ImportError:
        from irreptables import IrrepTable  # type: ignore

    from irrep.spacegroup_irreps import SpaceGroupIrreps
    
    cell = (primitive_atoms.cell, primitive_atoms.scaled_positions, primitive_atoms.numbers)
    sg = SpaceGroupIrreps.from_cell(
        cell=cell,
        spinor=False,
        include_TR=False,
        search_cell=True,
        symprec=symprec,
        verbosity=0
    )
    
    table = IrrepTable(sg.number_str, spinor=False)
    refUCTinv = np.linalg.inv(sg.refUC.T)
    
    seen_labels = set()
    results = []
    
    for irr in table.irreps:
        if irr.kpname not in seen_labels:
            seen_labels.add(irr.kpname)
            k_bcs = np.array(irr.k, dtype=float)
            k_input = refUCTinv @ k_bcs
            
            # Clean up near-zero values
            k_input = np.where(np.abs(k_input) < 1e-5, 0.0, k_input)
            k_bcs = np.where(np.abs(k_bcs) < 1e-5, 0.0, k_bcs)
            
            results.append({
                "label": irr.kpname,
                "qpoint_bcs": k_bcs.tolist(),
                "qpoint_input": k_input.tolist()
            })
            
    return results


def get_all_irreps_phonopy(
    phonopy_params,
    symprec: float | None = None,
    degeneracy_tolerance: float = 1e-4,
    log_level: int = 0,
) -> dict[str, IrRepsPhonopy]:
    """
    Compute irreps for all special q-points using direct phonopy calculations.
    Returns a dictionary mapping the q-point label to its IrRepsPhonopy instance.
    """
    phonon = phonopy_load(phonopy_params)
    primitive = phonon.primitive
    if symprec is None:
        symprec = phonon._symprec
        
    special_qs = get_special_qpoints(primitive, symprec=symprec)
    results = {}
    
    for sq in special_qs:
        label = sq["label"]
        q_input = sq["qpoint_input"]
        
        irr = IrRepsPhonopy(
            phonopy_params=phonopy_params,
            qpoint=q_input,
            symprec=symprec,
            degeneracy_tolerance=degeneracy_tolerance,
            log_level=log_level
        )
        irr.run(kpname=label)
        results[label] = irr
        
    return results

def find_highsym_qpoints_in_phbst(
    phbst_fname: str,
    symprec: float = 1e-5,
) -> list[dict]:
    """Find high-symmetry q-points that are present in a PHBST file.

    Reads the q-point list from the PHBST file, computes the theoretical
    high-symmetry points for the structure, and returns only those that
    match a q-point in the file.

    Returns:
        List of dicts with keys:
        - ind_q: index of the q-point in the PHBST file (0-based)
        - label: BCS label (e.g. 'GM', 'X')
        - qpoint: fractional coordinates in the input cell basis
    """
    atoms, qpoints, _freqs, _eig_vecs = read_phbst_freqs_and_eigvecs(phbst_fname)
    primitive = ase_atoms_to_phonopy_atoms(atoms)

    try:
        special_qs = get_special_qpoints(primitive, symprec=symprec)
    except Exception:
        special_qs = []

    matched = []
    for sq in special_qs:
        q_ref = np.array(sq["qpoint_input"])
        for ind_q, q in enumerate(qpoints):
            diff = np.array(q) - q_ref
            diff -= np.rint(diff)
            if np.all(np.abs(diff) < symprec):
                matched.append({
                    "ind_q": ind_q,
                    "label": sq["label"],
                    "qpoint": q.tolist(),
                })
                break  # each special point matched at most once

    return matched


def get_all_irreps_anaddb(
    phbst_fname: str,
    symprec: float = 1e-5,
    degeneracy_tolerance: float = 1e-4,
    log_level: int = 0,
) -> dict[int, IrRepsAnaddb]:
    """
    Compute irreps for all q-points present in the anaddb PHBST file.
    Returns a dictionary mapping the q-point index to its IrRepsAnaddb instance.
    """
    atoms, qpoints, freqs, eig_vecs = read_phbst_freqs_and_eigvecs(phbst_fname)
    primitive = ase_atoms_to_phonopy_atoms(atoms)
    
    try:
        special_qs = get_special_qpoints(primitive, symprec=symprec)
    except Exception:
        special_qs = []
        
    results = {}
    for ind_q, q in enumerate(qpoints):
        # Try to match q with a special q-point
        matched_label = None
        for sq in special_qs:
            q_input = np.array(sq["qpoint_input"])
            diff = q - q_input
            diff -= np.rint(diff)
            if np.all(np.abs(diff) < symprec):
                matched_label = sq["label"]
                break
                
        irr = IrRepsAnaddb(
            phbst_fname=phbst_fname,
            ind_q=ind_q,
            symprec=symprec,
            degeneracy_tolerance=degeneracy_tolerance,
            log_level=log_level
        )
        irr.run(kpname=matched_label)
        results[ind_q] = irr
        
    return results
