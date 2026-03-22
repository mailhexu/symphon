import numpy as np
from typing import Optional, List, cast
from scipy.constants import c, h, e, tera
from phonopy.phonon.irreps import IrReps, IrRepLabels
from phonopy.structure.symmetry import Symmetry
from phonopy.phonon.character_table import character_table
from ..io.phbst import read_phbst_freqs_and_eigvecs, ase_atoms_to_phonopy_atoms
from phonopy.phonon.degeneracy import degenerate_sets as get_degenerate_sets
from phonopy.structure.cells import is_primitive_cell
import spglib

from .reporting import ReportingMixin

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
        from .backend import IrRepsIrrep
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
        kpname = getattr(self, "_bcs_kpname", None) 
        # Actually in run() we didn't store kpname in self._bcs_kpname except implicitly in chiral transitions.
        # Let's just retrieve it from _irrep_backend_obj if available.
        if kpname is None and self._irrep_backend_obj and hasattr(self._irrep_backend_obj, "kpname"):
            kpname = self._irrep_backend_obj.kpname
        return (self._qpoint, kpname, indices)

    def _get_degenerate_sets(self):
        deg_sets = get_degenerate_sets(self._freqs, cutoff=self._degeneracy_tolerance)
        return deg_sets

    def _get_infrared_raman(self):
        """Compute IR- and Raman-active irreps using symmetry operations."""
        ir_active = set()
        raman_active = set()
        
        if self._pointgroup_symbol not in character_table:
            return ir_active, raman_active
            
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
