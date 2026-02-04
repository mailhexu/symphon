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
            ir_dict, raman_dict = raman_ir

            # IR: any non-None label associated with x, y, or z is IR-active.
            for lbl in ir_dict.values():
                if lbl:
                    ir_active_map[lbl] = True

            # Raman: labels present in any quadratic component.
            for labels in raman_dict.values():
                for lbl in labels:
                    if lbl:
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
        lines = []
        if summary:
            qx, qy, qz = summary[0]["qpoint"]
            lines.append(f"q-point: [{qx:.4f}, {qy:.4f}, {qz:.4f}]")
        point_group = getattr(self, "_pointgroup_symbol", None)
        if point_group:
            lines.append(f"Point group: {point_group}")
        if lines:
            lines.append("")
        if include_header:
            header = "# qx      qy      qz      band  freq(THz)   freq(cm-1)   label        IR  Raman"
            lines.append(header)
        for row in summary:
            qx, qy, qz = row["qpoint"]
            bi = row["band_index"]
            f_thz = row["frequency_thz"]
            f_cm1 = row["frequency_cm1"]
            label = row["label"] or "-"
            ir_flag = "Y" if row["is_ir_active"] else "."
            raman_flag = "Y" if row["is_raman_active"] else "."
            line = (
                f"{qx:7.4f} {qy:7.4f} {qz:7.4f}  {bi:4d}  "
                f"{f_thz:10.4f}  {f_cm1:11.2f}  {label:10s}  {ir_flag:^3s} {raman_flag:^5s}"
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

        # make symops in cartesian space
        rprim = self._primitive.cell  # np.identity(3)
        gprim = np.linalg.inv(rprim).T
        # print(" R G ", rprim, gprim)
        # print("rot ", self._rotations_at_q)

        # make cartesian symop matrices for each operation in each class
        # then get characters for IR and Raman reducible representations
        nclass = len(self._character_table["rotation_list"])
        self._cartesian_rotations_at_q = np.zeros([nclass, 96, 3, 3])
        degenclass = np.zeros(nclass)
        characters_xyz = np.zeros(nclass)
        chardegen_xyz = np.zeros(nclass)
        characters_x2 = np.zeros(nclass)
        chardegen_x2 = np.zeros(nclass)
        iclass = 0
        for opclass in self._character_table["mapping_table"].keys():
            degenclass[iclass] = len(self._character_table["mapping_table"][opclass])
            iop = 0
            for symop in np.array(self._character_table["mapping_table"][opclass][:]):
                # print("rotred ", symop)
                self._cartesian_rotations_at_q[iclass][iop] = np.dot(
                    rprim, np.dot(symop, gprim.T)
                )
                # print(" rotcart ", self._cartesian_rotations_at_q[isym])

                # m = self._cartesian_rotations_at_q[iclass][iop]
                # character_x2_iop =  np.matrix.trace(np.block([[m*m[0,0], m*m[0,1], m*m[0,2]],\
                #                                              [m*m[1,0], m*m[1,1], m*m[1,2]],\
                #                                              [m*m[2,0], m*m[2,1], m*m[2,2]]]))
                # print("class ", opclass, " op ", iop, " character ", character_x2_iop)

                iop += 1

            m = self._cartesian_rotations_at_q[iclass][0]
            # get representation characters for x,y,z functions
            characters_xyz[iclass] = np.matrix.trace(m)

            # get representation characters for quadratic functions
            # line below is in x2 xy y2 xz yz z2 format
            bigmat = np.zeros([6, 6])
            ibig = 0
            for ixyz in range(3):
                for ixyz_prime in range(ixyz + 1):
                    outprod = np.ndarray.flatten(np.outer(m[:, ixyz], m[:, ixyz_prime]))
                    bigmat[ibig, :] = [
                        outprod[0],
                        outprod[1] + outprod[3],
                        outprod[4],
                        outprod[2] + outprod[6],
                        outprod[5] + outprod[7],
                        outprod[8],
                    ]

                    ibig += 1
            # print(" class ", iclass, opclass, " x2 matrix", bigmat)

            characters_x2[iclass] = np.matrix.trace(bigmat)
            # print(" class ", iclass, "x2 charac ", characters_x2[iclass])

            chardegen_xyz[iclass] = characters_xyz[iclass] * degenclass[iclass]
            chardegen_x2[iclass] = characters_x2[iclass] * degenclass[iclass]
            # print("xyz charac ", characters_xyz[iclass], " degen ", self._degenclass[iclass])
            iclass += 1

        # now we have red representations, project them into irreps
        # print("irrep  characters g = ", self._g)
        xyzlabels = ["x", "y", "z"]
        x2labels = ["x^2", "xy", "y^2", "xz", "yz", "z^2"]
        IR_dict = {"x": None, "y": None, "z": None}
        Raman_dict = {"x^2": [], "xy": [], "y^2": [], "xz": [], "yz": [], "z^2": []}

        # loop over irreducible representations
        i_ir = 0
        for irreplabel in self._character_table["character_table"].keys():
            # characters
            irr_char = self._character_table["character_table"][irreplabel]
            # l_n dimention of current irreps
            len_irr = irr_char[0]
            # number of ir modes here
            n_ir = int(np.dot(irr_char, chardegen_xyz) / self._g)
            # number of Raman modes here
            n_ram = int(np.dot(irr_char, chardegen_x2) / self._g)
            # print(irreplabel, " nir ", n_ir, " nram ", n_ram, " irchar ", irr_char)

            # find eigenvectors: are x y or z isolated in representation?
            # IR
            for ixyz in range(3):
                xyzvec = np.zeros(3)
                for iclass in range(len(self._character_table["mapping_table"].keys())):
                    opclass = list(self._character_table["mapping_table"].keys())[iclass]
                    degenclass = len(self._character_table["mapping_table"][opclass][:])
                    for iop in range(degenclass):
                        xyzvec += (
                            irr_char[iclass]
                            * self._cartesian_rotations_at_q[iclass][iop][ixyz, :]
                        )
                xyzvec *= len_irr / self._g
                if np.linalg.norm(xyzvec) > 1.0e-6:
                    IR_dict[xyzlabels[ixyz]] = irreplabel

            # find the irreps which contain each of the quadratic functions (not full
            # linear combination basis functions, but still)
            # Raman
            ibig = 0
            bigvec = np.zeros(6)
            for ixyz in range(3):
                for ixyz_prime in range(ixyz + 1):
                    x2vec = np.zeros(6)
                    # loop over all operations
                    for iclass in range(
                        len(self._character_table["mapping_table"].keys())
                    ):
                        opclass = list(self._character_table["mapping_table"].keys())[iclass]
                        degenclass = len(self._character_table["mapping_table"][opclass][:])
                        for iop in range(degenclass):
                            m = self._cartesian_rotations_at_q[iclass][iop]
                            outprod = np.ndarray.flatten(
                                np.outer(m[:, ixyz], m[:, ixyz_prime])
                            )
                            bigvec = np.array(
                                [
                                    outprod[0],
                                    outprod[1] + outprod[3],
                                    outprod[4],
                                    outprod[2] + outprod[6],
                                    outprod[5] + outprod[7],
                                    outprod[8],
                                ]
                            )
                            x2vec += irr_char[iclass] * bigvec

                    x2vec *= len_irr / self._g
                    if np.linalg.norm(x2vec) > 1.0e-6:
                        # print(x2labels[ibig], " belongs to ", irreplabel , " norm ",
                        #       np.linalg.norm(x2vec))
                        Raman_dict[x2labels[ibig]].append(irreplabel)

                    ibig += 1
            # loop over irreps
            i_ir += 1

        return IR_dict, Raman_dict

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
            ir_dict, raman_dict = raman_ir

            # IR: any non-None label associated with x, y, or z is IR-active.
            for lbl in ir_dict.values():
                if lbl:
                    ir_active_map[lbl] = True

            # Raman: labels present in any quadratic component.
            for labels in raman_dict.values():
                for lbl in labels:
                    if lbl:
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
        lines = []
        if summary:
            qx, qy, qz = summary[0]["qpoint"]
            lines.append(f"q-point: [{qx:.4f}, {qy:.4f}, {qz:.4f}]")
        point_group = getattr(self, "_pointgroup_symbol", None)
        if point_group:
            lines.append(f"Point group: {point_group}")
        if lines:
            lines.append("")
        if include_header:
            header = "# qx      qy      qz      band  freq(THz)   freq(cm-1)   label        IR  Raman"
            lines.append(header)
        for row in summary:
            qx, qy, qz = row["qpoint"]
            bi = row["band_index"]
            f_thz = row["frequency_thz"]
            f_cm1 = row["frequency_cm1"]
            label = row["label"] or "-"
            ir_flag = "Y" if row["is_ir_active"] else "."
            raman_flag = "Y" if row["is_raman_active"] else "."
            line = (
                f"{qx:7.4f} {qy:7.4f} {qz:7.4f}  {bi:4d}  "
                f"{f_thz:10.4f}  {f_cm1:11.2f}  {label:10s}  {ir_flag:^3s} {raman_flag:^5s}"
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
