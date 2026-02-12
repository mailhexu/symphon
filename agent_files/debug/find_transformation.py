#!/usr/bin/env python3
"""
Find the k-point transformation matrix for Pnma.

How to run:
    uv run python agent_files/debug/find_transformation.py
"""

import numpy as np

# Data from irreptables (refUC) and what irrep expects (primUC)
kpoints = {
    "GM": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),  # ✓ No transformation needed
    "R":  ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # ✓ No transformation needed
    "S":  ([0.5, 0.5, 0.0], [0.0, 0.5, 0.5]),  # ✗ Needs transformation
    "T":  ([0.0, 0.5, 0.5], [0.5, 0.0, 0.5]),  # ✗ Needs transformation
    "U":  ([0.5, 0.0, 0.5], [0.5, 0.5, 0.0]),  # ✗ Needs transformation
    "X":  ([0.5, 0.0, 0.0], [0.0, 0.5, 0.0]),  # ✗ Needs transformation
    "Y":  ([0.0, 0.5, 0.0], [0.0, 0.0, 0.5]),  # ✗ Needs transformation
    "Z":  ([0.0, 0.0, 0.5], [0.5, 0.0, 0.0]),  # ✗ Needs transformation
}

def main():
    print("Finding transformation matrix P such that k_primUC = P @ k_refUC")
    print()
    
    # Try simple permutation matrices
    permutations = [
        ("Identity", np.eye(3)),
        ("(0,1,2) -> (1,2,0)", np.array([[0,1,0], [0,0,1], [1,0,0]])),
        ("(0,1,2) -> (2,0,1)", np.array([[0,0,1], [1,0,0], [0,1,0]])),
        ("(0,1,2) -> (0,2,1)", np.array([[1,0,0], [0,0,1], [0,1,0]])),
        ("(0,1,2) -> (2,1,0)", np.array([[0,0,1], [0,1,0], [1,0,0]])),
        ("(0,1,2) -> (1,0,2)", np.array([[0,1,0], [1,0,0], [0,0,1]])),
    ]
    
    for name, P in permutations:
        print(f"Testing {name}:")
        print(f"  P = {P.tolist()}")
        
        all_match = True
        for kpname, (k_ref, k_prim_expected) in kpoints.items():
            k_ref = np.array(k_ref)
            k_prim_expected = np.array(k_prim_expected)
            k_prim_computed = P @ k_ref
            
            matches = np.allclose(k_prim_computed, k_prim_expected)
            if not matches:
                all_match = False
                print(f"    {kpname}: {k_ref} -> {k_prim_computed} (expected {k_prim_expected}) ✗")
        
        if all_match:
            print(f"  ✓ ALL k-points match!")
            print()
            print(f"Found transformation matrix:")
            print(f"  {P}")
            return P
        print()
    
    print("No simple permutation matrix works!")
    return None

if __name__ == "__main__":
    P = main()
