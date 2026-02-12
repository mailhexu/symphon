#!/usr/bin/env python3
"""
Debug k-point transformation issue for TmFeO3.

The error message says:
  the kpoint [0.5 0.5 0. ] does not correspond to the point S 
  ([0.5 0.5 0. ] in refUC / [0. 0.5 0.5] in primUC) in the table

This means irreptables returns k=[0.5, 0.5, 0.0] but the irrep backend
expects k=[0.0, 0.5, 0.5] for the S point.

How to run:
    uv run python agent_files/debug/debug_kpoint_transform.py
"""

from phonopy.cui.load import load as phonopy_load
from irrep.spacegroup_irreps import SpaceGroupIrreps
from irreptables import IrrepTable
import numpy as np

def main():
    # Load TmFeO3
    phonon = phonopy_load("examples/3_advanced_phonopy/TmFeO3_phonopy_params.yaml")
    primitive = phonon.primitive
    
    # Get space group
    cell = (primitive.cell, primitive.scaled_positions, primitive.numbers)
    sg = SpaceGroupIrreps.from_cell(
        cell=cell,
        spinor=False,
        include_TR=False,
        search_cell=True,
        symprec=1e-2,
        verbosity=0
    )
    
    print(f"Space group: {sg.name}")
    print(f"Space group number: {sg.number}")
    print()
    
    # Get k-points from irreptables
    table = IrrepTable(sg.number_str, spinor=False)
    print("High-symmetry k-points from irreptables:")
    seen = set()
    for irrep in table.irreps:
        if hasattr(irrep, 'kpname') and irrep.kpname:
            if irrep.kpname not in seen:
                print(f"  {irrep.kpname:4s}: k={irrep.k}")
                seen.add(irrep.kpname)
    print()
    
    # Try to understand the S point issue
    print("Testing S point:")
    k_from_table = [0.5, 0.5, 0.0]
    print(f"  k from irreptables: {k_from_table}")
    
    try:
        result = sg.get_irreps_from_table("S", k_from_table)
        print(f"  SUCCESS: Got {len(result)} irreps")
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        print()
        print("  The irrep backend expects primitive UC coordinates,")
        print("  but irreptables gives reference UC coordinates.")
        print()
        print("  Possible solutions:")
        print("  1. Transform k-point from refUC to primUC")
        print("  2. Get transformation matrix from sg")
        print("  3. Use phonopy backend for these points")

if __name__ == "__main__":
    main()
