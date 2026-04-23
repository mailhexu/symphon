import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ..irreps.projection import (
    prepare_little_group_ops, 
    calculate_phonon_representation_matrices, 
    solve_unitary_mapping, 
    match_block_to_table
)
from irreptables.irreps import IrrepTable
from irrep.spacegroup_irreps import SpaceGroupIrreps
from phonopy.phonon.degeneracy import degenerate_sets
from .sam import SAMCalculator

# Conversion factor: 1 THz = 33.35641 cm-1
THZ_TO_CM1 = 33.35641

@dataclass
class CircularMode:
    """Information about a circularly polarized phonon mode."""
    band_index: int
    frequency: float  # THz
    frequency_cm1: float # cm-1
    irrep_label: str
    dimension: int
    circularity: float  # Dimensionless scalar circularity (0 to 1)
    sam: np.ndarray     # Spin Angular Momentum 3-vector
    displacement: np.ndarray # (N, 3) complex displacements
    opd: Optional[str] = None
    is_circular: bool = True
    is_elliptical: bool = False
    handedness: str = "N/A"  # "R", "L", or "N/A"

class CircularPhononFinder:
    """
    Finder for circularly polarized phonons.
    Identifies multi-dimensional IRs and constructs circular bases.
    """
    def __init__(self, primitive, qpoint, freqs, eigvecs, symprec=1e-5, log_level=0):
        self.primitive = primitive
        self.qpoint = np.array(qpoint)
        self.freqs = freqs
        self.eigvecs = eigvecs
        self.symprec = symprec
        self.log_level = log_level
        self._results = []

    def run(self, kpname: Optional[str] = None):
        """Run the identification and calculation."""
        # 1. Setup space group objects
        cell = (self.primitive.cell, self.primitive.scaled_positions, self.primitive.numbers)
        sg_obj = SpaceGroupIrreps.from_cell(
            cell=cell,
            spinor=False,
            include_TR=False,
            search_cell=True,
            symprec=self.symprec,
            verbosity=self.log_level,
        )
        spg_number = int(sg_obj.number_str.split('.')[0])
        table = IrrepTable(str(spg_number), sg_obj.spinor)
        
        # 2. Resolve k-point name
        if kpname is None:
            kpname = self._resolve_kpname(sg_obj, table)
        
        # 3. Get degenerate blocks
        deg_sets = degenerate_sets(self.freqs, cutoff=1e-4)
        
        # 4. Prepare symmetry ops
        sym_ops_processed, little_group_indices, _ = prepare_little_group_ops(
            self.primitive, self.qpoint, sg_obj.symmetries, sg_obj, table.symmetries, self.symprec
        )
        
        # 5. Calculate representation matrices
        is_gamma = (np.abs(self.qpoint) < self.symprec).all()
        block_matrices = calculate_phonon_representation_matrices(
            self.primitive, self.qpoint, sym_ops_processed, deg_sets, self.eigvecs,
            phase_convention='r', use_bcs_frame=(not is_gamma), sg_metadata=sg_obj
        )
        
        # 6. Process each block
        irreps_in_table = [irr for irr in table.irreps if irr.kpname == kpname]
        lg_bcs_info = self._build_lg_bcs_info(sg_obj, table, little_group_indices)
        
        for i_block, block in enumerate(deg_sets):
            dim = len(block)
            chars_calc = np.trace(block_matrices[i_block], axis1=1, axis2=2)
            matched_labels, total_dim = match_block_to_table(irreps_in_table, dim, chars_calc, lg_bcs_info)
            
            if matched_labels and total_dim == dim:
                # We have a labeled block. For circular phonons, we care about dim >= 2
                if dim >= 2:
                    self._process_multidim_block(sg_obj, little_group_indices, block, block_matrices[i_block], matched_labels)
                else:
                    # Record 1D (linear or elliptical) mode
                    for i_mode in range(dim):
                        b_idx = block[i_mode]
                        v_lin = self.eigvecs[:, b_idx].reshape(-1, 3)
                        sam_vec = self.calculate_sam(v_lin)
                        circ = SAMCalculator.get_circularity(sam_vec)
                        
                        is_elliptical = circ > 1e-4
                        handedness = "N/A"
                        if is_elliptical:
                            handedness = "R" if np.sum(sam_vec) > 0 else "L"

                        self._results.append(CircularMode(
                            band_index=b_idx + 1,
                            frequency=self.freqs[b_idx],
                            frequency_cm1=self.freqs[b_idx] * THZ_TO_CM1,
                            irrep_label="+".join(matched_labels),
                            dimension=dim,
                            circularity=circ,
                            sam=sam_vec,
                            displacement=v_lin,
                            opd="(1)",
                            is_circular=False,
                            is_elliptical=is_elliptical,
                            handedness=handedness
                        ))
        
        return self._results

    def _resolve_kpname(self, sg, table):
        refUC = sg.refUC
        q_bcs = refUC.T @ self.qpoint
        refUCTinv = np.linalg.inv(refUC.T)
        for irr in table.irreps:
            k_prim_table = refUCTinv @ irr.k
            diff = k_prim_table - self.qpoint
            if np.allclose(diff - np.round(diff), 0, atol=1e-4):
                return irr.kpname
        return "GM" if (np.abs(self.qpoint) < 1e-5).all() else None

    def _build_lg_bcs_info(self, sg, table, little_group_indices):
        refUC = sg.refUC
        refUCinv = np.linalg.inv(refUC)
        shiftUC = sg.shiftUC
        q_bcs = refUC.T @ self.qpoint
        
        rot_to_bcs = {}
        for i_tab, sym_tab in enumerate(table.symmetries):
            rot_to_bcs[tuple(sym_tab.R.flatten())] = (i_tab + 1, sym_tab.t % 1.0)
            
        lg_bcs_info = []
        for idx in little_group_indices:
            sym = sg.symmetries[idx]
            rot_bcs = np.round(refUCinv @ sym.rotation @ refUC).astype(int)
            trans_bcs_sg = np.round(refUCinv @ (sym.translation + sym.rotation @ shiftUC - shiftUC), 10) % 1.0
            rot_key = tuple(rot_bcs.flatten())
            if rot_key in rot_to_bcs:
                bcs_idx, t_tab = rot_to_bcs[rot_key]
                dT = trans_bcs_sg - t_tab
                dT = dT - np.round(dT)
                phase_corr = np.exp(2j * np.pi * np.dot(q_bcs, dT))
                lg_bcs_info.append((bcs_idx, phase_corr))
            else:
                lg_bcs_info.append((-1, 1.0))
        return lg_bcs_info

    def _process_multidim_block(self, sg, little_group_indices, block, block_mats, labels):
        """Construct circular modes for a multi-dimensional IR block."""
        from ..irreps.projection import get_reference_matrices, get_combined_reference_matrices
        
        # 1. Get reference matrices
        if len(labels) == 1:
            ref_mats, sp_lg_idx = get_reference_matrices(sg, labels[0], self.qpoint, None, self.symprec, self.log_level)
        else:
            ref_mats, sp_lg_idx = get_combined_reference_matrices(sg, labels, self.qpoint, None, self.symprec, self.log_level)
            
        if ref_mats is None or sp_lg_idx is None:
            return

        # 2. Match spgrep ops to block_mats (might need re-calc over sp_lg_idx)
        # For circular basis, we need the mapping U: Ref = U M U^\dagger => Ref U = U M
        # We need block_mats over the same indices as ref_mats
        sym_ops_sp, _, _ = prepare_little_group_ops(
            self.primitive, self.qpoint, [sg.symmetries[idx] for idx in sp_lg_idx], sg, None, self.symprec
        )
        is_gamma = (np.abs(self.qpoint) < self.symprec).all()
        block_mats_sp = calculate_phonon_representation_matrices(
            self.primitive, self.qpoint, sym_ops_sp, [block], self.eigvecs,
            phase_convention='r', use_bcs_frame=(not is_gamma), sg_metadata=sg
        )[0]
        
        U = solve_unitary_mapping(ref_mats, block_mats_sp)
        if U is None:
            return
            
        # U maps calculated eigvecs to IR basis: v_IR = U @ v_calc
        # The calculated eigvecs for this block are self.eigvecs[:, block]
        block_eigvecs = self.eigvecs[:, block] # (3N, dim)
        
        # Transform to aligned IR basis
        # v_aligned = sum_j U_ij v_calc,j
        aligned_eigvecs = block_eigvecs @ U.T # (3N, dim)
        
        # 3. Create circular combinations
        dim = len(block)
        if dim == 2:
            combinations = [
                (np.array([1, 1j]) / np.sqrt(2), "(1, i)"),
                (np.array([1, -1j]) / np.sqrt(2), "(1, -i)")
            ]
        elif dim == 3:
            combinations = [
                (np.array([1, 1j, 0]) / np.sqrt(2), "(1, i, 0)"),
                (np.array([1, -1j, 0]) / np.sqrt(2), "(1, -i, 0)"),
                (np.array([1, 0, 1j]) / np.sqrt(2), "(1, 0, i)"),
                (np.array([1, 0, -1j]) / np.sqrt(2), "(1, 0, -i)"),
                (np.array([0, 1, 1j]) / np.sqrt(2), "(0, 1, i)"),
                (np.array([0, 1, -1j]) / np.sqrt(2), "(0, 1, -i)")
            ]
        else:
            combinations = []
            
        for coeffs, opd in combinations:
            v_circ = aligned_eigvecs @ coeffs 
            v_circ_reshaped = v_circ.reshape(-1, 3)
            sam_vec = self.calculate_sam(v_circ_reshaped)
            circ = SAMCalculator.get_circularity(sam_vec)
            
            handedness = "R" if np.sum(sam_vec) > 0 else "L"
            if circ < 1e-4: handedness = "N/A"

            self._results.append(CircularMode(
                band_index=block[0] + 1, 
                frequency=self.freqs[block[0]],
                frequency_cm1=self.freqs[block[0]] * THZ_TO_CM1,
                irrep_label="+".join(labels),
                dimension=dim,
                circularity=circ,
                sam=sam_vec,
                displacement=v_circ_reshaped,
                opd=opd,
                is_circular=True,
                handedness=handedness
            ))
            
        # Also record the linear bases
        for i in range(dim):
            v_lin = aligned_eigvecs[:, i]
            v_lin_reshaped = v_lin.reshape(-1, 3)
            sam_vec = self.calculate_sam(v_lin_reshaped)
            circ = SAMCalculator.get_circularity(sam_vec)
            
            is_elliptical = circ > 1e-4
            handedness = "N/A"
            if is_elliptical:
                handedness = "R" if np.sum(sam_vec) > 0 else "L"

            self._results.append(CircularMode(
                band_index=block[i] + 1,
                frequency=self.freqs[block[i]],
                frequency_cm1=self.freqs[block[i]] * THZ_TO_CM1,
                irrep_label="+".join(labels),
                dimension=dim,
                circularity=circ,
                sam=sam_vec,
                displacement=v_lin_reshaped,
                opd=f"base_{i+1}",
                is_circular=False,
                is_elliptical=is_elliptical,
                handedness=handedness
            ))

    def calculate_sam(self, displacement):
        """
        Calculate Spin Angular Momentum 3-vector using Zhang-Niu formula.
        """
        return SAMCalculator.calculate(displacement, masses=self.primitive.masses)
