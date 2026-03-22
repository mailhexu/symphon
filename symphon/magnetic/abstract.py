import numpy as np
from typing import List, Dict, Any
import spglib
from spgrep import get_spacegroup_irreps_from_primitive_symmetry
from spgrep_modulation.isotropy import IsotropyEnumerator
from symphon.chiral import ChiralTransitionFinder
from symphon.symmetry_identification import (
    get_supercell_matrix_from_qpoint,
    get_lattice_translations_for_supercell,
)

class AbstractMagneticTransitionFinder:
    """
    Find abstract magnetic chiral phase transitions using pure group theory.
    This does not require an input structure, only the parent space group number.
    It assumes the transition is driven by a generic time-odd (magnetic) order parameter.
    """
    def __init__(self, spg_number: int, symprec: float = 1e-5):
        self.spg_number = spg_number
        self.symprec = symprec
        self._struct_finder = ChiralTransitionFinder(spg_number, symprec)
        self.spacegroup_info = self._struct_finder.spacegroup_info
        
        if self.spacegroup_info.is_sohncke:
            raise ValueError(f"Parent space group {spg_number} is already chiral (Sohncke group).")
        
    def find_transitions(self, qpoint: List[float], include_multi_k: bool = False) -> List[dict]:
        info = self.spacegroup_info
        P, P_inv = self._struct_finder._get_transformation_matrices()
        qpoint_prim = np.dot(P, qpoint)
        
        irreps, mapping = get_spacegroup_irreps_from_primitive_symmetry(
            info.primitive_rotations,
            info.primitive_translations,
            qpoint_prim
        )
        
        rots_lg = info.primitive_rotations[mapping]
        trans_lg = info.primitive_translations[mapping]
        
        import math
        k_conv = np.dot(P_inv, qpoint_prim)
        S_prim = get_supercell_matrix_from_qpoint(k_conv)
        # For multi-k: update denoms by LCM across all k-points in the star
        denoms = list(np.diag(S_prim).astype(int))
        S_prim_inv = np.linalg.inv(S_prim)
        lattice = np.dot(S_prim, info.primitive_lattice)
        
        M = max(denoms) + 2
        lattice_trans_n = get_lattice_translations_for_supercell(S_prim_inv, M)

        results = []
        
        # 1-k transitions
        for i, rep in enumerate(irreps):
            try:
                enumerator = IsotropyEnumerator(rots_lg, trans_lg, qpoint_prim, rep)
            except Exception:
                continue
                
            for opd in enumerator.order_parameter_directions:
                opd = opd.flatten()
                if len(opd) > rep.shape[1]: continue
                
                sc_rots, sc_trans, sc_time = [], [], []
                
                for j in range(len(rots_lg)):
                    r = rots_lg[j]
                    t = trans_lg[j]
                    mat_j = rep[j]
                    
                    for n in lattice_trans_n:
                        phase = np.exp(-2j * np.pi * np.dot(qpoint_prim, n))
                        mat = mat_j * phase
                        transformed = np.dot(mat, opd)
                        
                        r_prime = np.dot(S_prim_inv, np.dot(r, S_prim))
                        t_prime = np.dot(S_prim_inv, t + n)
                        
                        if np.linalg.norm(transformed - opd) < self.symprec:
                            sc_rots.append(r_prime); sc_trans.append(t_prime); sc_time.append(0)
                        elif np.linalg.norm(transformed + opd) < self.symprec:
                            sc_rots.append(r_prime); sc_trans.append(t_prime); sc_time.append(1)

                if sc_rots:
                    self._eval_and_append(sc_rots, sc_trans, sc_time, lattice, i, rep.shape[1], opd, "1-k", results)
                    
        # Multi-k transitions
        if include_multi_k:
            for i, rep in enumerate(irreps):
                try:
                    star, full_reps, subgroups = self._struct_finder._enumerate_multi_k_isotropy_subgroups(qpoint_prim, rep, mapping)
                except Exception:
                    continue
                    
                dim_small = rep.shape[1]
                
                for subgroup_indices, opd, _ in subgroups:
                    opd = opd.flatten()
                    sc_rots, sc_trans, sc_time = [], [], []
                    
                    for j in range(len(info.primitive_rotations)):
                        r = info.primitive_rotations[j]
                        t = info.primitive_translations[j]
                        mat_j = full_reps[j]
                        if mat_j is None: continue
                        
                        for n in lattice_trans_n:
                            phase_mat = np.zeros_like(mat_j, dtype=complex)
                            for idx_star, k_vec in enumerate(star):
                                phase = np.exp(-2j * np.pi * np.dot(k_vec, n))
                                s_idx = idx_star * dim_small
                                phase_mat[s_idx:s_idx+dim_small, s_idx:s_idx+dim_small] = np.eye(dim_small) * phase
                                
                            mat = np.dot(phase_mat, mat_j)
                            transformed = np.dot(mat, opd)
                            
                            r_prime = np.dot(S_prim_inv, np.dot(r, S_prim))
                            t_prime = np.dot(S_prim_inv, t + n)
                            
                            if np.linalg.norm(transformed - opd) < self.symprec:
                                sc_rots.append(r_prime); sc_trans.append(t_prime); sc_time.append(0)
                            elif np.linalg.norm(transformed + opd) < self.symprec:
                                sc_rots.append(r_prime); sc_trans.append(t_prime); sc_time.append(1)
                                
                    if sc_rots:
                        self._eval_and_append(sc_rots, sc_trans, sc_time, lattice, i, rep.shape[1], opd, f"{len(star)}-k", results)

        # Deduplicate
        seen = set()
        unique_results = []
        for res in results:
            key = (res['irrep_index'], res['bns_number'])
            if key not in seen:
                seen.add(key)
                unique_results.append(res)
                
        return unique_results

    def _eval_and_append(self, sc_rots, sc_trans, sc_time, lattice, irrep_idx, irrep_dim, opd, k_type, results):
        temp_sc_rots_arr = np.round(sc_rots).astype('intc')
        temp_sc_trans_arr = np.array(sc_trans, dtype='double')
        temp_sc_time_arr = np.array(sc_time, dtype='intc')
        
        try:
            msg_type = spglib.get_magnetic_spacegroup_type_from_symmetry(
                temp_sc_rots_arr, temp_sc_trans_arr, temp_sc_time_arr, 
                lattice=lattice, symprec=self.symprec
            )
            
            if msg_type is not None:
                is_chiral = True
                for r_op in temp_sc_rots_arr:
                    if np.linalg.det(r_op) < 0:
                        is_chiral = False
                        break
                        
                results.append({
                    'irrep_index': irrep_idx,
                    'irrep_dim': irrep_dim,
                    'k_type': k_type,
                    'opd': np.round(opd, 3).tolist(),
                    'uni_number': msg_type.uni_number,
                    'bns_number': msg_type.bns_number,
                    'is_chiral': is_chiral
                })
        except Exception:
            pass

