import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from spglib import get_symmetry, get_magnetic_symmetry_dataset, get_magnetic_spacegroup_type
from spgrep import get_spacegroup_irreps
from spgrep_modulation.isotropy import IsotropyEnumerator

class MagneticTransitionFinder:
    def __init__(self, cell: Tuple[np.ndarray, List[List[float]], List[int]], 
                 magnetic_sites: List[int], symprec: float = 1e-5):
        self.cell = cell
        self.lattice, self.positions, self.numbers = cell
        self.magnetic_sites = magnetic_sites
        self.symprec = symprec
        
    def find_magnetic_irreps(self, qpoint: List[float]) -> Tuple[Dict[int, dict], np.ndarray, np.ndarray]:
        """
        Finds the symmetry-adapted magnetic basis vectors for the given crystal.
        """
        irreps, rots_full, trans_full, mapping = get_spacegroup_irreps(
            self.lattice, self.positions, self.numbers, qpoint, symprec=self.symprec
        )
        
        rots_lg = rots_full[mapping]
        trans_lg = trans_full[mapping]
        
        num_ops = len(rots_lg)
        num_mag_atoms = len(self.magnetic_sites)
        
        mag_reps = []
        for i in range(num_ops):
            R = rots_lg[i]
            t = trans_lg[i]
            mat = np.zeros((num_mag_atoms * 3, num_mag_atoms * 3), dtype=complex)
            axial_R = np.linalg.det(R) * R
            
            for j, atom_idx in enumerate(self.magnetic_sites):
                pos_j = self.positions[atom_idx]
                new_pos = np.dot(R, pos_j) + t
                
                for k, map_idx in enumerate(self.magnetic_sites):
                    diff = new_pos - self.positions[map_idx]
                    diff -= np.round(diff)
                    if np.linalg.norm(diff) < self.symprec:
                        translation_diff = new_pos - self.positions[map_idx]
                        phase = np.exp(-2j * np.pi * np.dot(qpoint, translation_diff))
                        mat[k*3:(k+1)*3, j*3:(j+1)*3] = phase * axial_R
                        break
            mag_reps.append(mat)
            
        mag_reps = np.array(mag_reps)
        
        basis_dict = {}
        for i, irrep_mats in enumerate(irreps):
            dim = irrep_mats.shape[1]
            chars = np.trace(irrep_mats, axis1=1, axis2=2)
            
            P = np.zeros((num_mag_atoms * 3, num_mag_atoms * 3), dtype=complex)
            for g in range(num_ops):
                P += np.conj(chars[g]) * mag_reps[g]
            P *= dim / num_ops
            
            u, s, vh = np.linalg.svd(P)
            rank = np.sum(s > 1e-4)
            
            if rank > 0:
                basis_dict[i] = {
                    'dim': dim,
                    'rank': rank,
                    'mats': irrep_mats,
                    'basis': u[:, :rank]
                }
                
        return basis_dict, rots_lg, trans_lg

    def build_supercell(self, qpoint: List[float], opd: np.ndarray, basis: np.ndarray) -> Tuple[Tuple, bool]:
        """
        Builds the commensurately modulated magnetic supercell.
        Supports q-points with denominator 1, 2, 3, 4, etc.
        """
        q = np.array(qpoint)
        # Find denominator
        from fractions import Fraction
        denoms = [Fraction(x).limit_denominator(10).denominator for x in q]
        lcm_denom = np.lcm.reduce(denoms)
        
        sc_matrix = np.array([
            [lcm_denom if q[0] != 0 else 1, 0, 0],
            [0, lcm_denom if q[1] != 0 else 1, 0],
            [0, 0, lcm_denom if q[2] != 0 else 1]
        ])
        
        sc_lattice = self.lattice @ sc_matrix
        
        sc_positions = []
        sc_numbers = []
        sc_magmoms = []
        
        opd = opd.flatten()
        dim = len(opd)
        
        # If irrep appears multiple times, basis has shape (3N, m*dim).
        # We only need one copy (the first 'dim' columns) to break symmetry correctly.
        basis_dim = basis[:, :dim]
        base_spin = np.dot(basis_dim, opd).reshape(len(self.magnetic_sites), 3)
        
        for nx in range(sc_matrix[0,0]):
            for ny in range(sc_matrix[1,1]):
                for nz in range(sc_matrix[2,2]):
                    t_vec = np.array([nx, ny, nz])
                    phase = np.exp(2j * np.pi * np.dot(q, t_vec))
                    
                    for i, pos in enumerate(self.positions):
                        sc_pos = (pos + t_vec) / np.diag(sc_matrix)
                        sc_positions.append(sc_pos)
                        sc_numbers.append(self.numbers[i])
                        
                        if i in self.magnetic_sites:
                            mag_idx = self.magnetic_sites.index(i)
                            spin_c = base_spin[mag_idx] * phase
                            spin = 2 * np.real(spin_c)
                            sc_magmoms.append(spin)
                        else:
                            sc_magmoms.append([0.0, 0.0, 0.0])
                            
        sc_magmoms = np.array(sc_magmoms)
        sc_magmoms[np.abs(sc_magmoms) < 1e-4] = 0.0
        
        is_zero = np.allclose(sc_magmoms, 0)
        return (sc_lattice, sc_positions, sc_numbers, sc_magmoms), is_zero

    def check_chirality(self, dataset) -> bool:
        """A magnetic group is chiral if all symmetries are proper rotations (det=1)."""
        if dataset is None:
            return False
        for r in dataset.rotations:
            if np.linalg.det(r) < 0:
                return False
        return True

    def find_transitions(self, qpoint: List[float]) -> List[dict]:
        """
        Main pipeline to find all chiral magnetic transitions at a q-point.
        """
        basis_dict, rots_lg, trans_lg = self.find_magnetic_irreps(qpoint)
        results = []
        
        for irrep_idx, info in basis_dict.items():
            enumerator = IsotropyEnumerator(
                little_rotations=rots_lg,
                little_translations=trans_lg,
                qpoint=np.array(qpoint),
                small_rep=info['mats']
            )
            
            for opd in enumerator.order_parameter_directions:
                if len(opd.shape) > 1 and opd.shape[0] > 1:
                    continue # Skip continuous OPDs for now
                    
                sc_cell, is_zero = self.build_supercell(qpoint, opd, info['basis'])
                if is_zero:
                    continue
                    
                msg_dataset = get_magnetic_symmetry_dataset(sc_cell, symprec=self.symprec)
                if msg_dataset is None:
                    continue
                    
                msg_type = get_magnetic_spacegroup_type(msg_dataset.uni_number)
                is_chiral = self.check_chirality(msg_dataset)
                
                results.append({
                    'irrep_index': irrep_idx,
                    'irrep_dim': info['dim'],
                    'opd': opd.flatten().tolist(),
                    'uni_number': msg_dataset.uni_number,
                    'bns_number': msg_type.bns_number,
                    'is_chiral': is_chiral,
                })
                
        return results
