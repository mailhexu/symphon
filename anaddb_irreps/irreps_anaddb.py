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

#        if not self._is_primitive_cell():
#            print('')
#            print("Non-primitve cell is used.")
#            print("Your unit cell may be transformed to a primitive cell "
#                  "by PRIMITIVE_AXIS tag.")
#            return False
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
            #print (" char tab ", self._character_table)

            if (abs(self._q) < self._symprec).all() and self._rotation_symbols:
                self._ir_labels = self._get_ir_labels()
                self._RamanIR_labels = self._get_infrared_raman()
                IR_labels, Ram_labels = self._RamanIR_labels
                print ("IR labels ", IR_labels)
                print ("Ram labels ", Ram_labels)
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

    def _get_infrared_raman(self):
        """once we have irreps and characters, use them and symops to find the IR and Raman active irreps
        http://symmetry.jacobs-university.de/cgi-bin/group.cgi?group=606&option=4
        """
# make symops in cartesian space
        rprim = self._primitive.cell  #np.identity(3)
        gprim = np.linalg.inv(rprim).T
        #print (" R G ", rprim, gprim)
        #print ("rot ", self._rotations_at_q)

# make cartesian symop matrices for each operation in each class
# then get characters for IR and Raman reducible representations
        nclass = len(self._character_table['rotation_list'])
        self._cartesian_rotations_at_q = np.zeros([nclass,96,3,3])
        degenclass = np.zeros(nclass)
        characters_xyz = np.zeros(nclass)
        chardegen_xyz = np.zeros(nclass)
        characters_x2 = np.zeros(nclass)
        chardegen_x2 = np.zeros(nclass)
        iclass = 0
        for opclass in self._character_table['mapping_table'].keys():
          degenclass[iclass] = len(self._character_table['mapping_table'][opclass])
          iop = 0
          for symop in  np.array(self._character_table['mapping_table'][opclass][:]):
            #print ("rotred ", symop)
            self._cartesian_rotations_at_q[iclass][iop] = np.dot(rprim, np.dot(symop, gprim.T))
            #print (" rotcart ", self._cartesian_rotations_at_q[isym])

            #m = self._cartesian_rotations_at_q[iclass][iop]
            #character_x2_iop =  np.matrix.trace(np.block([[m*m[0,0], m*m[0,1], m*m[0,2]],\
            #                                              [m*m[1,0], m*m[1,1], m*m[1,2]],\
            #                                              [m*m[2,0], m*m[2,1], m*m[2,2]]]))
            #print ("class ", opclass, " op ", iop, " character ", character_x2_iop)

            iop += 1 

          m = self._cartesian_rotations_at_q[iclass][0]
# get representation characters for x,y,z functions
          characters_xyz[iclass] = np.matrix.trace(m)

# get representation characters for quadratic functions
# line below is in x2 xy y2 xz yz z2 format
          bigmat = np.zeros([6,6])
          ibig = 0
          for ixyz in range(3):
            for ixyz_prime in range(ixyz+1):
              outprod = np.ndarray.flatten(np.outer(m[:,ixyz],m[:,ixyz_prime]))
              bigmat[ibig,:] = [outprod[0], \
                 outprod[1]+outprod[3], outprod[4], \
                 outprod[2]+outprod[6], outprod[5]+outprod[7], outprod[8]]
                
              ibig += 1
          #print (" class ", iclass, opclass, " x2 matrix", bigmat)

          characters_x2[iclass] = np.matrix.trace(bigmat)
          #print (" class ", iclass, "x2 charac ", characters_x2[iclass])

          chardegen_xyz[iclass] = characters_xyz[iclass]*degenclass[iclass]
          chardegen_x2[iclass]  = characters_x2[iclass]*degenclass[iclass]
          #print ("xyz charac ", characters_xyz[iclass], " degen ", self._degenclass[iclass])
          iclass += 1

# now we have red representations, project them into irreps
        #print ("irrep  characters g = ", self._g)
        xyzlabels = ['x','y','z']
        x2labels = ['x^2', 'xy', 'y^2', 'xz', 'yz', 'z^2']
        IR_dict={'x':None, 'y':None, 'z':None}
        Raman_dict={'x^2':[],'xy':[],'y^2':[],'xz':[],'yz':[],'z^2':[]}

# loop over irreducible representations
        i_ir = 0
        for irreplabel in self._character_table['character_table'].keys():
          # characters
          irr_char = self._character_table['character_table'][irreplabel]
          # l_n dimention of current irreps
          len_irr = irr_char[0]
          # number of ir modes here
          n_ir = int(np.dot(irr_char,chardegen_xyz)/self._g)
          # number of Raman modes here
          n_ram = int(np.dot(irr_char,chardegen_x2)/self._g)
          #print (irreplabel, " nir ", n_ir, " nram ", n_ram, " irchar ", irr_char)

# find eigenvectors: are x y or z isolated in representation?
# IR
          for ixyz in range(3):
            xyzvec = np.zeros(3)
            for iclass in range(len(self._character_table['mapping_table'].keys())):
              opclass = list(self._character_table['mapping_table'].keys())[iclass]
              degenclass = len(self._character_table['mapping_table'][opclass][:])
              for iop in range(degenclass):
                xyzvec += irr_char[iclass] * self._cartesian_rotations_at_q[iclass][iop][ixyz,:]
            xyzvec *= len_irr/self._g
            if np.linalg.norm(xyzvec) > 1.e-6:
              IR_dict[xyzlabels[ixyz]] = irreplabel

# find the irreps which contain each of the quadratic functions (not full linear combination basis functions, but still)
# Raman
          ibig = 0
          bigvec = np.zeros(6)
          for ixyz in range(3):
            for ixyz_prime in range(ixyz+1):
              x2vec = np.zeros(6)
# loop over all operations
              for iclass in range(len(self._character_table['mapping_table'].keys())):
                opclass = list(self._character_table['mapping_table'].keys())[iclass]
                degenclass = len(self._character_table['mapping_table'][opclass][:])
                for iop in range(degenclass):
                  m = self._cartesian_rotations_at_q[iclass][iop]
                  outprod = np.ndarray.flatten(np.outer(m[:,ixyz],m[:,ixyz_prime]))
                  bigvec = np.array([outprod[0], \
                     outprod[1]+outprod[3], outprod[4], \
                     outprod[2]+outprod[6], outprod[5]+outprod[7], outprod[8]])
                  x2vec  += irr_char[iclass] * bigvec

              x2vec *= len_irr/self._g
              if np.linalg.norm(x2vec) > 1.e-6:
                #print (x2labels[ibig], " belongs to ", irreplabel , " norm ", np.linalg.norm(x2vec))
                Raman_dict[x2labels[ibig]].append(irreplabel)

              ibig += 1
# loop over irreps
          i_ir += 1

        return IR_dict, Raman_dict

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
