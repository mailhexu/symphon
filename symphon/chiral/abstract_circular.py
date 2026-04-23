import numpy as np
from typing import List, Dict, Any, Optional
from irreptables.irreps import IrrepTable
from .sohncke import is_sohncke

class AbstractCircularPhononFinder:
    """
    Identifies potential circular phonons purely from group theory.
    Scans a space group's special k-points for multidimensional IRs.
    """
    def __init__(self, spg_number: int):
        self.spg_number = spg_number
        self.table = IrrepTable(str(spg_number), spinor=False)

    def find_candidates(self) -> List[Dict[str, Any]]:
        """
        Find k-points and IRs that support circular polarization.
        """
        results = []
        # Group irreps by k-point
        kpoints = {}
        for irr in self.table.irreps:
            if irr.kpname not in kpoints:
                kpoints[irr.kpname] = {
                    'name': irr.kpname,
                    'coords': irr.k,
                    'irreps': []
                }
            kpoints[irr.kpname]['irreps'].append(irr)

        for kpname, kp_data in kpoints.items():
            multidim_irreps = [irr for irr in kp_data['irreps'] if irr.dim > 1]
            if multidim_irreps:
                results.append({
                    'kpname': kpname,
                    'kcoords': kp_data['coords'],
                    'candidates': [
                        {
                            'name': irr.name,
                            'dim': irr.dim,
                            'possible_opds': self._get_possible_opds(irr.dim)
                        } for irr in multidim_irreps
                    ]
                })
        
        return results

    def _get_possible_opds(self, dim: int) -> List[str]:
        if dim == 2:
            return ["(1, i)", "(1, -i)"]
        if dim == 3:
            return ["(1, i, 0)", "(1, 0, i)", "(0, 1, i)", "(1, w, w^2)"]
        return []

def get_all_sohncke_candidates() -> Dict[int, List[Dict[str, Any]]]:
    """Scan all 65 Sohncke space groups for circular phonon candidates."""
    all_results = {}
    for i in range(1, 231):
        if is_sohncke(i):
            finder = AbstractCircularPhononFinder(i)
            cands = finder.find_candidates()
            if cands:
                all_results[i] = cands
    return all_results
