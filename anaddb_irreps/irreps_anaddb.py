import numpy as np
from ase import Atoms
from phonopy.phonon.irreps import IrReps, IrRepLabels
from phonopy.structure.symmetry import Symmetry
from phonopy.phonon.character_table import character_table
from anaddb_irreps.abipy_io import read_phbst_freqs_and_eigvecs, ase_atoms_to_phonopy_atoms
from phonopy.phonon.degeneracy import degenerate_sets as get_degenerate_sets
from phonopy.structure.cells import is_primitive_cell
from phonopy import load as phonopy_load


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
        conv = 33.35641  # 1 THz -> cm^-1
        n_modes = len(freqs_thz)

        irreps = getattr(self, "_irreps", None)

        # Build IR/Raman activity maps from _RamanIR_labels when available.
        ir_active_map: dict = {}
        raman_active_map: dict = {}

        raman_ir = getattr(self, "_RamanIR_labels", None)
        if raman_ir is not None:
            ir_labels, raman_labels = raman_ir
            for lbl in ir_labels:
                ir_active_map[lbl] = True
            for lbl in raman_labels:
                raman_active_map[lbl] = True

        # Extract labels using degenerate sets and _ir_labels when using phonopy backend.
        # If _irreps is already list of dicts (irrep backend), we use it directly.
        raw_labels = [None] * n_modes
        ir_labels_seq = getattr(self, "_ir_labels", None)
        deg_sets = getattr(self, "_degenerate_sets", None)

        # Build a mapping from mode index to degenerate set index
        mode_to_degset = {}
        if deg_sets is not None:
            for set_idx, deg_set in enumerate(deg_sets):
                for mode_idx in deg_set:
                    mode_to_degset[mode_idx] = set_idx

        for band_index in range(n_modes):
            label = None

            # 1) Prefer label attached to irreps[band_index] when present.
            if irreps is not None and band_index < len(irreps):
                ir = irreps[band_index]
                if hasattr(ir, "label"):
                    label = ir.label
                elif isinstance(ir, dict) and "label" in ir:
                    label = ir["label"]

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

            raw_labels[band_index] = label

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

            summary.append(
                {
                    "qpoint": q,
                    "band_index": band_index,
                    "frequency_thz": freq_thz,
                    "frequency_cm1": freq_cm1,
                    "label": label,
                    "is_ir_active": is_ir_active,
                    "is_raman_active": is_raman_active,
                }
            )

        return summary

    def format_summary_table(self, include_header: bool = True) -> str:
        """Format the summary table as a human-readable string."""
        summary = self.get_summary_table()

        # Only show activity columns if we have activity data (supported by backend)
        backend = getattr(self, "_backend", "phonopy")
        show_activity = backend == "phonopy"
        
        # For both_labels mode, show extra BCS label column
        show_both = getattr(self, "_both_labels", False) and getattr(self, "_irrep_labels_both", None) is not None

        lines = []
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
            if show_activity and show_both:
                header = "# qx      qy      qz      band  freq(THz)   freq(cm-1)   label(M)   label(BCS)  IR  Raman"
            elif show_activity:
                header = "# qx      qy      qz      band  freq(THz)   freq(cm-1)   label        IR  Raman"
            else:
                header = "# qx      qy      qz      band  freq(THz)   freq(cm-1)   label"
            lines.append(header)

        for i, row in enumerate(summary):
            qx, qy, qz = row["qpoint"]
            bi = row["band_index"]
            f_thz = row["frequency_thz"]
            f_cm1 = row["frequency_cm1"]
            label = row["label"] or "-"
            
            if show_both:
                # Get both labels: phonopy (Mulliken) and irrep (BCS)
                irrep_labels_both = getattr(self, "_irrep_labels_both", None)
                label_mulliken = label if label != "-" else ""
                label_bcs = irrep_labels_both[i] if i < len(irrep_labels_both) else ""
                
                if show_activity:
                    ir_flag = "Y" if row["is_ir_active"] else "."
                    raman_flag = "Y" if row["is_raman_active"] else "."
                    line = (
                        f"{qx:7.4f} {qy:7.4f} {qz:7.4f}  {bi:4d}  "
                        f"{f_thz:10.4f}  {f_cm1:11.2f}  {label_mulliken:10s}  {label_bcs:10s}  {ir_flag:^3s} {raman_flag:^5s}"
                    )
                else:
                    line = (
                        f"{qx:7.4f} {qy:7.4f} {qz:7.4f}  {bi:4d}  "
                        f"{f_thz:10.4f}  {f_cm1:11.2f}  {label_mulliken:10s}  {label_bcs:10s}"
                    )
            elif show_activity:
                ir_flag = "Y" if row["is_ir_active"] else "."
                raman_flag = "Y" if row["is_raman_active"] else "."
                line = (
                    f"{qx:7.4f} {qy:7.4f} {qz:7.4f}  {bi:4d}  "
                    f"{f_thz:10.4f}  {f_cm1:11.2f}  {label:10s}  {ir_flag:^3s} {raman_flag:^5s}"
                )
            else:
                line = (
                    f"{qx:7.4f} {qy:7.4f} {qz:7.4f}  {bi:4d}  "
                    f"{f_thz:10.4f}  {f_cm1:11.2f}  {label:10s}"
                )
            lines.append(line)

        return "\n".join(lines)

    def get_verbose_output(self) -> str:
        """Get verbose phonopy-style output (only for phonopy backend)."""
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
        is_little_cogroup: bool = False,
        symprec: float = 1e-5,
        degeneracy_tolerance: float = 1e-5,
        log_level: int = 0,
        backend: str = "phonopy",
        both_labels: bool = False,
    ) -> None:
        self._is_little_cogroup = is_little_cogroup
        self._symprec = symprec
        self._degeneracy_tolerance = degeneracy_tolerance
        self._log_level = log_level

        self._qpoint = np.array(qpoint)
        self._degeneracy_tolerance = degeneracy_tolerance
        self._primitive = primitive_atoms
        self._freqs, self._eig_vecs = freqs, eigvecs
        self._character_table = None
        self._verbose = False
        self._backend = backend.lower()
        self._backend_obj = None
        self._both_labels = both_labels
        self._irrep_labels_both = None
        self._irrep_backend_obj = None


class IrRepsAnaddb(IrRepsEigen):
    """Irreps helper tied to anaddb PHBST output."""

    def __init__(
        self,
        phbst_fname,
        ind_q,
        is_little_cogroup: bool = False,
        symprec: float = 1e-5,
        degeneracy_tolerance: float = 1e-5,
        log_level: int = 0,
        backend: str = "phonopy",
    ) -> None:
        atoms, qpoints, freqs, eig_vecs = read_phbst_freqs_and_eigvecs(phbst_fname)
        primitive_atoms = ase_atoms_to_phonopy_atoms(atoms)

        super().__init__(
            primitive_atoms,
            qpoints[ind_q],
            freqs[ind_q],
            eig_vecs[ind_q],
            is_little_cogroup=is_little_cogroup,
            symprec=symprec,
            degeneracy_tolerance=degeneracy_tolerance,
            log_level=log_level,
            backend=backend,
        )


def print_irreps(
    phbst_fname,
    ind_q,
    is_little_cogroup=False,
    symprec=1e-5,
    degeneracy_tolerance=1e-4,
    log_level=0,
    show_verbose=False,
    backend="phonopy",
    kpname=None,
):
    irr = IrRepsAnaddb(
        phbst_fname=phbst_fname,
        ind_q=ind_q,
        is_little_cogroup=is_little_cogroup,
        symprec=symprec,
        degeneracy_tolerance=degeneracy_tolerance,
        log_level=log_level,
        backend=backend,
    )
    irr.run(kpname=kpname)

    # Print summary table
    print(irr.format_summary_table())

    # Optionally print verbose output
    if show_verbose and backend == "phonopy":
        print()
        print("# Verbose irreps output")
        print(irr.get_verbose_output())

    return irr


def print_irreps_phonopy(
    phonopy_params,
    qpoint,
    is_little_cogroup: bool = False,
    symprec: float | None = None,
    degeneracy_tolerance: float = 1e-4,
    log_level: int = 0,
    show_verbose: bool = False,
    backend: str = "phonopy",
    kpname=None,
    both_labels: bool = False,
):
    irr = IrRepsPhonopy(
        phonopy_params=phonopy_params,
        qpoint=qpoint,
        is_little_cogroup=is_little_cogroup,
        symprec=symprec,
        degeneracy_tolerance=degeneracy_tolerance,
        log_level=log_level,
        backend=backend,
        both_labels=both_labels,
    )
    irr.run(kpname=kpname)

    # Print summary table
    print(irr.format_summary_table())

    # Optionally print verbose output
    if show_verbose and backend == "phonopy":
        print()
        print("# Verbose irreps output")
        print(irr.get_verbose_output())

    return irr
