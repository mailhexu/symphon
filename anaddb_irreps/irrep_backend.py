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
        # If kpname is not provided, try to infer it from the qpoint
        if kpname is None:
            # Fallback to Gamma if close to zero
            if (np.abs(self._qpoint) < self._symprec).all():
                kpname = "GM"
            else:
                raise ValueError("kpname (e.g., 'GM', 'X', 'M') must be provided for non-Gamma points when using 'irrep' backend.")
        
        try:
            bcs_table = sg.get_irreps_from_table(kpname, self._qpoint)
        except Exception as e:
            if self._log_level > 0:
                print(f"Error fetching BCS table for {kpname} at {self._qpoint}: {e}")
            raise e

        # 3. Calculate phonon traces
        all_traces, little_group = self._calculate_phonon_traces(sg)
        
        # 4. Match traces with BCS labels
        self._irreps = []
        for block in self._degenerate_sets:
            # Calculate total trace for the degenerate block (reducible representation of the block)
            block_traces_sum = np.sum(all_traces[block, :], axis=0)
            
            best_match = None
            max_overlap = 0
            
            for label, table_chars in bcs_table.items():
                g = len(table_chars) # Size of the little group in the table
                overlap = 0
                for i, sym in enumerate(little_group):
                    if sym.ind in table_chars:
                        # Standard projection formula: multiplicity n = 1/g * sum(chi_irrep* * chi_block)
                        overlap += np.conj(table_chars[sym.ind]) * block_traces_sum[i]
                
                n = overlap / g
                match_val = np.abs(n)
                
                if match_val > max_overlap:
                    max_overlap = match_val
                    best_match = label
            
            # Store result for each mode in the block
            for _ in block:
                # If multiplicity is close to 1 (or integer), it's a good match
                if max_overlap > 0.8:
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
                
        num_little = len(little_group_indices)
        num_modes = self._eigvecs.shape[1]
        traces = np.zeros((num_modes, num_little), dtype=complex)
        
        L = self._primitive.cell
        Linv = np.linalg.inv(L)

        for idx, isym in enumerate(little_group_indices):
            sym = sg.symmetries[isym]
            rot = sym.rotation
            trans = sym.translation
            
            # Cartesian rotation for polar vectors
            R_cart = L.T @ rot @ Linv.T
            
            perm = []
            phases = []
            for k in range(num_atoms):
                new_pos = rot @ positions[k] + trans
                found = False
                for j in range(num_atoms):
                    diff = new_pos - positions[j]
                    diff_round = np.round(diff)
                    if np.allclose(diff - diff_round, 0, atol=1e-5):
                        perm.append(j)
                        phases.append(np.exp(-2j * np.pi * np.dot(self._qpoint, diff_round)))
                        found = True
                        break
                if not found:
                    raise RuntimeError(f"Atom mapping failed for sym {sym.ind}")

            for m in range(num_modes):
                ev = self._eigvecs[:, m].reshape(num_atoms, 3)
                tr = 0
                for k in range(num_atoms):
                    j = perm[k]
                    tr += np.dot(ev[j].conj(), R_cart @ ev[k]) * phases[k]
                traces[m, idx] = tr
                
        return traces, [sg.symmetries[i] for i in little_group_indices]
