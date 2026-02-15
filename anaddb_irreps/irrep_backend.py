import numpy as np
from phonopy.phonon.degeneracy import degenerate_sets
from ase.data import atomic_masses
import logging
import spglib

try:
    from irrep.spacegroup_irreps import SpaceGroupIrreps
    HAS_IRREP = True
except ImportError:
    HAS_IRREP = False

class IrRepsIrrep:
    """
    Backend driver using the 'irrep' package for irrep identification.
    """
    def __init__(self, primitive, qpoint, freqs, eigvecs, symprec=1e-5, log_level=0):
        if not HAS_IRREP:
            raise ImportError("The 'irrep' package is required for this backend. Install it with 'pip install irrep'.")
        
        self._primitive = primitive # PhonopyAtoms
        self._qpoint = np.array(qpoint)
        self._freqs = freqs
        self._eigvecs = eigvecs # (3N, 3N)
        self._symprec = symprec
        self._log_level = log_level
        
        self._irreps = None
        self._degenerate_sets = None
        self._pointgroup_symbol = None
        self._spacegroup_symbol = None
        self._RamanIR_labels = None # Not implemented for non-gamma in this backend yet
        self._bcs_kpname = None  # Store BCS k-point label
        self._qpoint_bcs = None  # Store k-point in BCS coordinates
        
    def run(self, kpname=None):
        # 1. Setup SpaceGroupIrreps from irrep package
        cell = (self._primitive.cell, self._primitive.scaled_positions, self._primitive.numbers)
        sg = SpaceGroupIrreps.from_cell(
            cell=cell,
            spinor=False,
            include_TR=False,
            search_cell=True,
            symprec=self._symprec,
            verbosity=self._log_level
        )
        
        self._spacegroup_symbol = sg.name
        self._degenerate_sets = degenerate_sets(self._freqs)
        
        # Get point group from symmetry operations
        rotations = np.array([sym.rotation for sym in sg.symmetries], dtype=int)
        pg_result = spglib.get_pointgroup(rotations)
        if pg_result:
            self._pointgroup_symbol = pg_result[0]
        else:
            self._pointgroup_symbol = None
        
        # 2. Get BCS character table for the q-point
        # Use refUC to find the matching label in the BCS table
        from irreptables import IrrepTable
        table = IrrepTable(sg.number_str, sg.spinor)
        refUC = sg.refUC
        refUCTinv = np.linalg.inv(refUC.T)
        
        # Transform q-point to BCS coordinates
        q_bcs = refUC.T @ self._qpoint
        self._qpoint_bcs = q_bcs
        
        found_kpname = None
        for irr in table.irreps:
            k_prim_table = (refUCTinv @ irr.k)
            diff = (k_prim_table - self._qpoint)
            if np.allclose(diff - np.round(diff), 0, atol=1e-4):
                found_kpname = irr.kpname
                break
        
        # Print coordinate mapping information
        if self._log_level > 0:
            print(f"\nK-point Coordinate Mapping:")
            print(f"  Primitive: [{self._qpoint[0]:.4f}, {self._qpoint[1]:.4f}, {self._qpoint[2]:.4f}]")
            print(f"  BCS:       [{q_bcs[0]:.4f}, {q_bcs[1]:.4f}, {q_bcs[2]:.4f}]")
            if found_kpname:
                print(f"  BCS Label: {found_kpname}")
        
        if kpname is not None and found_kpname is not None and kpname != found_kpname:
            if self._log_level > 0:
                print(f"Warning: Provided kpname '{kpname}' differs from detected BCS label '{found_kpname}'. Using '{found_kpname}'.")
        
        if found_kpname:
            kpname = found_kpname
            self._bcs_kpname = found_kpname
        elif kpname is None:
            # Fallback to Gamma if close to zero
            if (np.abs(self._qpoint) < self._symprec).all():
                kpname = "GM"
            else:
                raise ValueError(f"Could not identify BCS label for q-point {self._qpoint}. Please provide kpname.")
        
        try:
            # Pass original qpoint, irrep handles transformation internally
            bcs_table = sg.get_irreps_from_table(kpname, self._qpoint)
        except Exception as e:
            if self._log_level > 0:
                print(f"Error fetching BCS table for {kpname} at {self._qpoint}: {e}")
            raise e

        # 3. Detect if BCS table has only multi-dimensional irreps
        # Check dimension of each irrep (identity character = dimension)
        min_irrep_dim = float('inf')
        for label, table_chars in bcs_table.items():
            if 1 in table_chars:
                identity_char = table_chars[1]
            else:
                identity_char = list(table_chars.values())[0]
            dim = int(round(abs(identity_char)))
            min_irrep_dim = min(min_irrep_dim, dim)
        
        only_multidim = (min_irrep_dim > 1)
        
        if only_multidim and self._log_level > 0:
            print(f"\nDetected k-point with ONLY {min_irrep_dim}D irreps (no 1D irreps)")
            print(f"Force-pairing consecutive modes into {min_irrep_dim}D blocks...")
        
        # 4. Apply force-pairing if needed
        if only_multidim and min_irrep_dim == 2:
            # Force pair consecutive modes for 2D irreps
            original_deg_sets = self._degenerate_sets
            forced_pairs = []
            i = 0
            while i < len(self._freqs):
                if i + 1 < len(self._freqs):
                    # Pair consecutive modes
                    forced_pairs.append(tuple([i, i+1]))
                    i += 2
                else:
                    # Odd number of modes - leave last one as single
                    forced_pairs.append(tuple([i]))
                    i += 1
            
            self._degenerate_sets = forced_pairs
            
            if self._log_level > 0:
                print(f"  Original degeneracy: {len(original_deg_sets)} blocks")
                print(f"  Forced pairing: {len(forced_pairs)} blocks")
                singles = sum(1 for b in forced_pairs if len(b) == 1)
                pairs = sum(1 for b in forced_pairs if len(b) == 2)
                print(f"    Singles: {singles}, Pairs: {pairs}")
        
        # 5. Calculate phonon traces
        all_traces, little_group_indices = self._calculate_phonon_traces(sg)
        
        # 6. Match traces with BCS labels
        # Note: BCS table keys are 1-indexed positions in sg.symmetries (full list)
        # We stored indices in little_group_indices, add 1 for 1-based indexing
        #
        # IMPORTANT: Phonon eigenvectors have arbitrary phases (gauge freedom).
        # The BCS irrep tables assume a specific phase convention that may differ
        # from phonopy's convention, especially at non-Gamma q-points.
        # This can cause incorrect irrep identification at zone boundary points.
        #
        # The matching uses the projection formula:
        # n = (1/g) * sum_g conj(chi_table(g)) * chi_block(g)
        # where n should be close to 1 for a good match.
        
        self._irreps = []
        for block in self._degenerate_sets:
            block_size = len(block)
            
            # Build representation matrices for each symmetry in little group
            block_traces = []
            for idx in range(len(little_group_indices)):
                if block_size == 1:
                    block_traces.append(all_traces[block[0], idx])
                else:
                    block_traces.append(np.sum([all_traces[m, idx] for m in block]))
            
            best_match = None
            max_overlap = 0
            all_overlaps = {}
            
            for label, table_chars in bcs_table.items():
                g = len(table_chars)
                overlap = 0
                for idx, isym in enumerate(little_group_indices):
                    char_key = isym + 1
                    if char_key in table_chars:
                        overlap += np.conj(table_chars[char_key]) * block_traces[idx]
                
                n = overlap / g
                match_val = np.abs(n)
                all_overlaps[label] = match_val
                
                if match_val > max_overlap:
                    max_overlap = match_val
                    best_match = label
            
            # Debug output for unidentified modes
            # Use adaptive threshold based on k-point
            is_gamma = (np.abs(self._qpoint) < self._symprec).all()
            threshold = 0.8 if is_gamma else 0.5
            
            if self._log_level > 2 and max_overlap < threshold:
                freq = self._freqs[block[0]]
                print(f"  Block (modes {list(block)}, freq={freq:.4f}): best={best_match} overlap={max_overlap:.4f}")
                sorted_overlaps = sorted(all_overlaps.items(), key=lambda x: -x[1])[:3]
                print(f"    Top overlaps: {sorted_overlaps}")
                print(f"    Block traces: {[f'{t:.3f}' for t in block_traces]}")
            
            # Store result for each mode in the block
            for _ in block:
                # Threshold considerations:
                # - Gamma point: overlap ~1.0 for correct identification, use threshold 0.8
                # - Non-Gamma points: overlap typically 0.5-0.7 due to eigenvector gauge mismatch, use threshold 0.5
                # - Eigenvector phases from phonopy may not match BCS convention at non-Gamma k-points
                # - Lower threshold at non-Gamma enables identification but increases false positives
                # - Users should verify non-Gamma assignments, especially those with overlap < 0.6
                if max_overlap > threshold:
                    self._irreps.append({"label": best_match})
                else:
                    self._irreps.append({"label": None})
                    
        return True

    def _calculate_phonon_traces(self, sg):
        num_atoms = len(self._primitive.scaled_positions)
        positions = self._primitive.scaled_positions
        
        # Identify little group of q
        little_group_indices = []
        for i, sym in enumerate(sg.symmetries):
            dq = np.dot(sym.rotation, self._qpoint) - self._qpoint
            if np.allclose(dq - np.round(dq), 0, atol=1e-5):
                little_group_indices.append(i)
        
        if self._log_level > 1:
            print(f"  Little group size: {len(little_group_indices)}")
        
        # Check if we're at Gamma
        is_gamma = (np.abs(self._qpoint) < self._symprec).all()
        
        # Transformation parameters for BCS frame
        refUC = sg.refUC
        shiftUC = sg.shiftUC
        refUCinv = np.linalg.inv(refUC)
        
        # At Gamma, use primitive frame directly (no BCS transformation needed)
        # At non-Gamma points, transform to BCS frame for consistent phases
        if is_gamma:
            positions_work = positions
            positions_work_unmodded = positions
            q_work = self._qpoint
            use_bcs_frame = False
        else:
            # Transform positions to BCS frame
            positions_bcs_unmodded = np.array([(refUCinv @ (p - shiftUC)) for p in positions])
            positions_work = positions_bcs_unmodded % 1
            positions_work_unmodded = positions_bcs_unmodded
            q_work = refUC.T @ self._qpoint
            use_bcs_frame = True
                
        num_little = len(little_group_indices)
        num_modes = self._eigvecs.shape[1]
        traces = np.zeros((num_modes, num_little), dtype=complex)
        
        L = self._primitive.cell # (lattice vectors as rows)
        Linv = np.linalg.inv(L)
        
        if use_bcs_frame:
            # BCS Cartesian frame transformation
            L_bcs = L @ refUC
            L_bcs_inv = np.linalg.inv(L_bcs)
        else:
            # Use primitive frame directly at Gamma
            L_bcs = L
            L_bcs_inv = Linv

        for idx, isym in enumerate(little_group_indices):
            sym = sg.symmetries[isym]
            rot_prim = sym.rotation
            trans_prim = sym.translation
            
            if use_bcs_frame:
                # Symmetry in BCS frame
                rot_work = np.round(refUCinv @ rot_prim @ refUC).astype(int)
                trans_work = refUCinv @ (trans_prim + rot_prim @ shiftUC - shiftUC)
            else:
                # Use primitive frame directly at Gamma
                rot_work = rot_prim
                trans_work = trans_prim
            
            # Cartesian rotation
            R_cart = L_bcs.T @ rot_work @ L_bcs_inv.T
            
            perm = []
            phases = []
            for k in range(num_atoms):
                new_pos = rot_work @ positions_work[k] + trans_work
                found = False
                for j in range(num_atoms):
                    diff = new_pos - positions_work[j]
                    diff_round = np.round(diff)
                    if np.allclose(diff - diff_round, 0, atol=self._symprec):
                        perm.append(j)
                        # r-gauge phase factor
                        L_vec = rot_work @ positions_work_unmodded[k] + trans_work - positions_work_unmodded[k]
                        phase = np.exp(2j * np.pi * np.dot(q_work, L_vec))
                        phases.append(phase)
                        found = True
                        break
                if not found:
                    raise RuntimeError(f"Atom mapping failed for sym {sym.ind}")

            for m in range(num_modes):
                ev = self._eigvecs[:, m].reshape(num_atoms, 3)
                
                if use_bcs_frame:
                    # Transform eigenvectors to BCS Cartesian frame (axis permutation/rotation)
                    ev_work = (refUCinv @ ev.T).T
                else:
                    # Use eigenvectors directly at Gamma
                    ev_work = ev
                
                tr = 0
                for k in range(num_atoms):
                    j = perm[k]
                    tr += np.dot(ev_work[j].conj(), R_cart @ ev_work[k]) * phases[k]
                traces[m, idx] = tr
                
        return traces, little_group_indices
