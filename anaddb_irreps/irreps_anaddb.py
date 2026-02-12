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
            if show_activity:
                header = "# qx      qy      qz      band  freq(THz)   freq(cm-1)   label        IR  Raman"
            else:
                header = "# qx      qy      qz      band  freq(THz)   freq(cm-1)   label"
            lines.append(header)

        for row in summary:
            qx, qy, qz = row["qpoint"]
            bi = row["band_index"]
            f_thz = row["frequency_thz"]
            f_cm1 = row["frequency_cm1"]
            label = row["label"] or "-"
            
            if show_activity:
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
    ) -> None:
        self._is_little_cogroup = is_little_cogroup
        self._log_level = log_level

        self._qpoint = np.array(qpoint)
        self._degeneracy_tolerance = degeneracy_tolerance
        self._symprec = symprec
        self._primitive = primitive_atoms
        self._freqs, self._eig_vecs = freqs, eigvecs
        self._character_table = None
        self._verbose = False
        self._backend = backend.lower()
        self._backend_obj = None

    def run(self, kpname=None) -> bool:
        if self._backend == "irrep":
            from .irrep_backend import IrRepsIrrep
            self._backend_obj = IrRepsIrrep(
                primitive=self._primitive,
                qpoint=self._qpoint,
                freqs=self._freqs,
                eigvecs=self._eig_vecs,
                symprec=self._symprec,
                log_level=self._log_level
            )
            res = self._backend_obj.run(kpname=kpname)
            # Sync attributes for ReportingMixin
            self._irreps = self._backend_obj._irreps
            self._degenerate_sets = self._backend_obj._degenerate_sets
            self._pointgroup_symbol = self._backend_obj._pointgroup_symbol
            self._spacegroup_symbol = self._backend_obj._spacegroup_symbol
            return res

        # Existing phonopy logic
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

        (self._transformation_matrix, self._conventional_rotations,) = self._get_conventional_rotations()

        self._ground_matrices = self._get_ground_matrix()
        self._degenerate_sets = self._get_degenerate_sets()
        self._irreps = self._get_irreps()
        self._characters, self._irrep_dims = self._get_characters()

        self._ir_labels = None

        if (
            self._pointgroup_symbol in character_table.keys()
            and character_table[self._pointgroup_symbol] is not None
        ):
            self._rotation_symbols, character_table_of_ptg = self._get_rotation_symbols(self._pointgroup_symbol)
            self._character_table = character_table_of_ptg
            # print(" char tab ", self._character_table)

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
        else:
            self._rotation_symbols = None
            if self._log_level > 0:
                print(f"Point group {self._pointgroup_symbol} not found in database.")

        return True

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
        is_little_cogroup: bool = False,
        symprec: float | None = None,
        degeneracy_tolerance: float = 1e-5,
        log_level: int = 0,
        backend: str = "phonopy",
    ) -> None:
        phonon = phonopy_load(phonopy_params)
        q = np.asarray(qpoint, dtype=float)
        phonon.run_qpoints([q], with_eigenvectors=True)
        q_dict = phonon.get_qpoints_dict()
        freqs = np.array(q_dict["frequencies"][0], dtype=float)
        eigvecs = np.array(q_dict["eigenvectors"][0], dtype=complex)
        primitive_atoms = phonon.primitive

        if symprec is None:
            inferred = None
            sym_obj = getattr(phonon, "_symmetry", None)
            if sym_obj is not None:
                inferred = getattr(sym_obj, "_symmetry_tolerance", None)
                if inferred is None:
                    inferred = getattr(sym_obj, "tolerance", None)
            if inferred is None:
                inferred = 1e-5
            symprec_value = float(inferred)
        else:
            symprec_value = float(symprec)

        super().__init__(
            primitive_atoms,
            q,
            freqs,
            eigvecs,
            is_little_cogroup=is_little_cogroup,
            symprec=symprec_value,
            degeneracy_tolerance=degeneracy_tolerance,
            log_level=log_level,
            backend=backend,
        )


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
):
    irr = IrRepsPhonopy(
        phonopy_params=phonopy_params,
        qpoint=qpoint,
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
