import numpy as np
import re
from typing import Optional, List
from scipy.constants import c, tera
from ..chiral.sohncke import is_sohncke, get_sohncke_class, SohnckeClass
from ..chiral.transitions import ChiralTransitionFinder

class ReportingMixin:
    """Mixin class for consistent reporting output."""

    def _build_ir_raman_maps(self):
        """Return (ir_active_map, raman_active_map) dicts from _RamanIR_labels."""
        ir_active_map: dict[str, bool] = {}
        raman_active_map: dict[str, bool] = {}
        raman_ir = getattr(self, "_RamanIR_labels", None)
        if raman_ir is not None:
            ir_labels, raman_labels = raman_ir
            for lbl in ir_labels:
                ir_active_map[lbl] = True
            for lbl in raman_labels:
                raman_active_map[lbl] = True
        return ir_active_map, raman_active_map

    def _collect_raw_labels_opds(self, n_modes, irreps, ir_labels_seq, deg_sets):
        """First pass: collect raw label and OPD for each band index.

        Returns (raw_labels, raw_opds) as lists of length n_modes.
        """
        raw_labels: List[Optional[str]] = [None] * n_modes
        raw_opds: List[Optional[str]] = [None] * n_modes

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
                set_idx = mode_to_degset.get(band_index)
                if set_idx is not None and set_idx < len(ir_labels_seq):
                    cand = ir_labels_seq[set_idx]
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

        return raw_labels, raw_opds

    def _propagate_labels_in_degsets(self, raw_labels, deg_sets):
        """Second pass: propagate labels within degenerate sets in-place."""
        if deg_sets is None:
            return
        for deg_set in deg_sets:
            labels_in_set = {raw_labels[i] for i in deg_set if raw_labels[i]}
            if len(labels_in_set) == 1:
                lbl = labels_in_set.pop()
                for i in deg_set:
                    raw_labels[i] = lbl

    def _build_daughter_str(self, band_index, label, opd_str, irreps, chiral_map):
        """Resolve daughter SG string and possibly update opd_str for one band.

        Returns (opd_str, daughter_str).
        """
        daughter_str = "-"

        # 4) Prefer daughter_sg from backend if present.
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

        # 5) Lookup chiral transitions if daughter_str is still "-".
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
                trans_list = chiral_map.get(lbl)
                if not trans_list:
                    trans_list = chiral_map.get(lbl.split(":")[0])
                if trans_list:
                    if not isinstance(trans_list, list):
                        trans_list = [trans_list]
                    if opd_str == "-":
                        opds = sorted(set(t.opd.symbolic for t in trans_list))
                        opd_str = ", ".join(opds)
                    daughters = sorted(set(
                        f"{t.daughter_spg_symbol}(#{t.daughter_spg_number})"
                        for t in trans_list
                    ))
                    daughter_str = ", ".join(daughters)
                    break

        return opd_str, daughter_str

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
        ir_labels_seq = getattr(self, "_ir_labels", None)
        deg_sets = getattr(self, "_degenerate_sets", None)
        chiral_map = getattr(self, "_chiral_transitions_map", {})

        ir_active_map, raman_active_map = self._build_ir_raman_maps()
        raw_labels, raw_opds = self._collect_raw_labels_opds(
            n_modes, irreps, ir_labels_seq, deg_sets
        )
        self._propagate_labels_in_degsets(raw_labels, deg_sets)

        summary = []
        for band_index, f_thz in enumerate(freqs_thz):
            freq_thz = float(f_thz)
            freq_cm1 = freq_thz * conv
            label = raw_labels[band_index]
            is_ir_active = bool(label and ir_active_map.get(label, False))
            is_raman_active = bool(label and raman_active_map.get(label, False))
            opd_str = raw_opds[band_index] or "-"
            opd_str, daughter_str = self._build_daughter_str(
                band_index, label, opd_str, irreps, chiral_map
            )
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

    @staticmethod
    def _align_supercell_labels(summary, supercell_labels):
        """Frequency-align supercell labels to band indices.

        Returns a list of length len(summary) where each entry is the matched
        daughter SG string from supercell_labels, or None if unmatched.
        """
        sc_label_by_band: list = [None] * len(summary)
        if not supercell_labels:
            return sc_label_by_band
        remaining = list(enumerate(supercell_labels))
        for band_i, row in enumerate(summary):
            if not remaining:
                break
            band_freq = row["frequency_thz"]
            best_j, best_dist = 0, abs(remaining[0][1][0] - band_freq)
            for j, (_, (gf, _)) in enumerate(remaining):
                d = abs(gf - band_freq)
                if d < best_dist:
                    best_dist, best_j = d, j
            _, (_, sc_sg) = remaining.pop(best_j)
            sc_label_by_band[band_i] = sc_sg
        return sc_label_by_band

    @staticmethod
    def _build_deg_groups(summary):
        """Return dict mapping band_i to list of band indices in its degenerate block."""
        deg_groups: dict = {}
        i = 0
        while i < len(summary):
            row = summary[i]
            lbl = row.get("label") or ""
            freq = row["frequency_thz"]
            j = i + 1
            while (j < len(summary)
                   and abs(summary[j]["frequency_thz"] - freq) < 1e-4
                   and (summary[j].get("label") or "") == lbl):
                j += 1
            group = list(range(i, j))
            for k in group:
                deg_groups[k] = group
            i = j
        return deg_groups

    @staticmethod
    def _chiral_class_suffix(dsg: str) -> str:
        """Return 'II-pair', 'III', or '-' for the Sohncke class of daughter SG."""
        if dsg == "-" or "(#" not in dsg:
            return "-"
        try:
            m = re.search(r"\(#(\d+)\)", dsg)
            if not m:
                return "-"
            sohncke_cls = get_sohncke_class(int(m.group(1)))
            if sohncke_cls == SohnckeClass.CLASS_II:
                return "II-pair"
            if sohncke_cls == SohnckeClass.CLASS_III:
                return "III"
            return "-"
        except (ValueError, IndexError):
            return "-"

    def _format_header(self, include_qpoint_cols, show_both, show_activity,
                       show_chiral_cols, show_supercell_labels):
        """Build and return the column header string."""
        if include_qpoint_cols:
            header = f"{'# qx':>8s} {'qy':>8s} {'qz':>8s} {'band':>5s} {'freq(THz)':>10s} {'freq(cm-1)':>11s}"
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

        if show_supercell_labels:
            header += f" {'SC Daughter SG':<25s} {'Match':<6s}"

        return header

    def _format_row(self, i, row, include_qpoint_cols, show_both, show_activity,
                    show_chiral_cols, show_supercell_labels,
                    sc_label_by_band, deg_groups, summary):
        """Build and return one data row string."""
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
            label_bcs = "-"
            if irrep_labels_bcs and i < len(irrep_labels_bcs):
                label_bcs = irrep_labels_bcs[i] or "-"
            line += f" {str(label):>12s} {str(label_bcs):>12s}"
        else:
            line += f" {str(label):>12s}"

        if show_activity:
            line += f" {ir_flag:^4s} {raman_flag:^6s}"

        if show_chiral_cols:
            opd = row.get("opd") or "-"
            dsg = row.get("daughter_sg") or "-"
            line += f" {str(opd):<15s} {str(dsg):<20s}"
            line += f" {self._chiral_class_suffix(dsg):<8s}"
        elif show_both:
            opd = row.get("opd") or "-"
            line += f" {str(opd):<15s}"

        if show_supercell_labels:
            sc_sg = sc_label_by_band[i]
            sc_str = sc_sg if sc_sg else "-"
            impl_sg = row.get("daughter_sg", "-")
            if impl_sg != "-" and sc_sg:
                if impl_sg == sc_sg:
                    match_str = "Y"
                else:
                    group = deg_groups.get(i, [i])
                    if len(group) > 1:
                        our_set = {summary[k].get("daughter_sg", "-") for k in group}
                        sc_set = {sc_label_by_band[k] for k in group if sc_label_by_band[k]}
                        match_str = "Y" if our_set == sc_set else "N"
                    else:
                        match_str = "N"
            else:
                match_str = "N"
            line += f" {sc_str:<25s} {match_str:<6s}"

        return line

    def format_summary_table(self, include_header: bool = True, include_symmetry: bool = True, include_qpoint_cols: bool = True, show_chiral: bool = False, supercell_labels=None) -> str:
        """Format the summary table as a human-readable string.

        Args:
            include_header: Whether to include column headers
            include_symmetry: Whether to include q-point, space group, and point group info
            include_qpoint_cols: Whether to include qx, qy, qz columns in the table
            show_chiral: Whether to include OPD and daughter SG chiral transition columns
            supercell_labels: Optional list of (freq, daughter_sg) from spgrep-modulation
                               supercell calculations, for side-by-side comparison.
        """
        summary = self.get_summary_table()
        show_chiral_cols = any(row.get("daughter_sg") != "-" for row in summary)
        show_supercell_labels = supercell_labels is not None

        sc_label_by_band = self._align_supercell_labels(summary, supercell_labels) if show_supercell_labels else [None] * len(summary)
        deg_groups = self._build_deg_groups(summary)

        bcs_labels = getattr(self, "_irrep_labels_bcs", [])
        bcs_opds = getattr(self, "_irrep_opds_bcs", [])
        show_both = any(lbl for lbl in bcs_labels) or any(opd for opd in bcs_opds)
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
            lines.append(self._format_header(
                include_qpoint_cols, show_both, show_activity,
                show_chiral_cols, show_supercell_labels,
            ))

        for i, row in enumerate(summary):
            lines.append(self._format_row(
                i, row, include_qpoint_cols, show_both, show_activity,
                show_chiral_cols, show_supercell_labels,
                sc_label_by_band, deg_groups, summary,
            ))

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
