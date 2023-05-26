import numpy as np
from ase import Atoms
from phonopy.phonon.irreps import IrReps
from phonopy.structure.symmetry import Symmetry
from phonopy.phonon.character_table import character_table
from anaddb_irreps.abipy_io import read_phbst_freqs_and_eigvecs, ase_atoms_to_phonopy_atoms
from phonopy.phonon.degeneracy import degenerate_sets as get_degenerate_sets
from phonopy.structure.cells import is_primitive_cell


class IrRepsEigen(IrReps):
    def __init__(
            self,
            primitive_atoms,
            q,
            freqs,
            eigvecs,
            is_little_cogroup=False,
            symprec=1e-5,
            degeneracy_tolerance=1e-5,
            log_level=0):
        self._is_little_cogroup = is_little_cogroup
        # self._nac_q_direction = nac_q_direction
        self._log_level = log_level

        self._q = np.array(q)
        self._degeneracy_tolerance = degeneracy_tolerance
        self._symprec = symprec
        self._primitive = primitive_atoms
        # self._primitive = dynamical_matrix.get_primitive()
        # self._dynamical_matrix = dynamical_matrix
        # self._ddm = DerivativeOfDynamicalMatrix(dynamical_matrix)
        self._freqs, self._eigvecs = freqs, eigvecs
        self._character_table = None

    def run(self):
        self._symmetry_dataset = Symmetry(self._primitive,
                                          symprec=self._symprec).get_dataset()

        # if not self._is_primitive_cell():
        #    print('')
        #    print("Non-primitve cell is used.")
        #    print("Your unit cell may be transformed to a primitive cell "
        #          "by PRIMITIVE_AXIS tag.")
        #    return False
        if not is_primitive_cell(self._symmetry_dataset["rotations"]):
            raise RuntimeError(
                "Non-primitve cell is used. Your unit cell may be transformed to "
                "a primitive cell by PRIMITIVE_AXIS tag."
            )

        (self._rotations_at_q,
         self._translations_at_q) = self._get_rotations_at_q()

        self._g = len(self._rotations_at_q)

        (self._pointgroup_symbol, self._transformation_matrix,
         self._conventional_rotations) = self._get_conventional_rotations()

        self._ground_matrices = self._get_ground_matrix()
        self._degenerate_sets = self._get_degenerate_sets()
        self._irreps = self._get_irreps()
        self._characters, self._irrep_dims = self._get_characters()

        self._ir_labels = None

        if (self._pointgroup_symbol in character_table.keys()
                and character_table[self._pointgroup_symbol] is not None):
            self._rotation_symbols = self._get_rotation_symbols()
            if (abs(self._q) < self._symprec).all() and self._rotation_symbols:
                self._ir_labels = self._get_ir_labels()
            elif (abs(self._q) < self._symprec).all():
                if self._log_level > 0:
                    print("Database for this point group is not preprared.")
            else:
                if self._log_level > 0:
                    print("Database for non-Gamma point is not prepared.")
        else:
            self._rotation_symbols = None

        return True

    def _get_degenerate_sets(self):
        deg_sets = get_degenerate_sets(self._freqs,
                                       cutoff=self._degeneracy_tolerance)
        return deg_sets


class IrRepsAnaddb(IrRepsEigen):
    def __init__(self,
                 phbst_fname,
                 ind_q,
                 is_little_cogroup=False,
                 symprec=1e-5,
                 degeneracy_tolerance=1e-5,
                 log_level=0):
        """
        phbst_fname: name of phbst file.
        ind_q: index of the q point in the phbst.
        is_little_cogroup: 
        symprec: precision for deciding the symmetry of the atomic structure.
        degeneracy_tolenrance: the tolerance of frequency difference in deciding the degeneracy.
        log_level: how much information is in the output. 
        """
        atoms, qpoints, freqs, eigvecs = read_phbst_freqs_and_eigvecs(
            phbst_fname)
        primitive_atoms = ase_atoms_to_phonopy_atoms(atoms)

        super().__init__(primitive_atoms,
                         qpoints[ind_q],
                         freqs[ind_q],
                         eigvecs[ind_q],

                         is_little_cogroup=is_little_cogroup,
                         symprec=symprec,
                         degeneracy_tolerance=degeneracy_tolerance,
                         log_level=log_level)
