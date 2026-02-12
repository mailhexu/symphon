#!/usr/bin/env python3
"""
Debug k-point transformation - find the transformation matrix.

How to run:
    uv run python agent_files/debug/debug_kpoint_transform2.py
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
    
    print(f"Space group: {sg.name} (#{sg.number})")
    print()
    
    # Check for transformation matrix
    print("Looking for transformation matrix in SpaceGroupIrreps:")
    for attr in dir(sg):
        if 'trans' in attr.lower() or 'matrix' in attr.lower() or 'prim' in attr.lower():
            try:
                val = getattr(sg, attr)
                if not callable(val):
                    print(f"  {attr}: {val}")
            except:
                pass
    print()
    
    # Try the S point with different coordinates
    print("Testing S point:")
    k_refUC = np.array([0.5, 0.5, 0.0])
    k_primUC = np.array([0.0, 0.5, 0.5])  # From error message
    
    print(f"  k (refUC):  {k_refUC}")
    print(f"  k (primUC): {k_primUC}")
    
    # Try with primUC coordinates
    try:
        result = sg.get_irreps_from_table("S", k_primUC)
        print(f"  SUCCESS with primUC coords: Got {len(result)} irreps")
    except RuntimeError as e:
        print(f"  FAILED with primUC coords: {e}")
    
    # Look for transformation pattern
    print()
    print("Checking transformation pattern:")
    print("  If k_primUC = P @ k_refUC, then:")
    print("  [0.0]   [?  ?  ?]   [0.5]")
    print("  [0.5] = [?  ?  ?] @ [0.5]")
    print("  [0.5]   [?  ?  ?]   [0.0]")
    print()
    print("  Possible permutation: (0,1,2) -> (2,0,1) or similar")
    
    # Test other points
    print()
    print("Testing other k-points:")
    tests = [
        ("T", [0.0, 0.5, 0.5]),
        ("U", [0.5, 0.0, 0.5]),
        ("X", [0.5, 0.0, 0.0]),
        ("Y", [0.0, 0.5, 0.0]),
        ("Z", [0.0, 0.0, 0.5]),
    ]
    
    for kpname, k in tests:
        try:
            result = sg.get_irreps_from_table(kpname, k)
            print(f"  {kpname}: k={k} ✓")
        except RuntimeError as e:
            # Extract the expected coordinates from error message
            msg = str(e)
            if "in primUC" in msg:
                prim_part = msg.split("in primUC)")[0].split("[")[-1]
                print(f"  {kpname}: k={k} ✗  (expects {prim_part})")

if __name__ == "__main__":
    main()
