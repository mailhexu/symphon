#!/usr/bin/env python
"""
CLI tool for finding chiral magnetic phase transitions.

Usage:
    magnetic-chiral --structure POSCAR --qpoint 0 0 0.5 --mag-sites 0,1
"""

import argparse
import sys
import numpy as np
from ase.io import read
from anaddb_irreps.magnetic_transitions import MagneticTransitionFinder

def parse_args():
    parser = argparse.ArgumentParser(description="Find chiral magnetic transitions")
    parser.add_argument("--structure", required=True, help="Path to structural file (e.g., POSCAR, CIF)")
    parser.add_argument("--qpoint", nargs=3, type=float, default=[0, 0, 0], help="Propagation wavevector (fractional)")
    parser.add_argument("--mag-sites", type=str, required=True, help="Comma-separated list of magnetic atom indices (0-based)")
    parser.add_argument("--symprec", type=float, default=1e-5, help="Symmetry precision")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Read structure using ASE
    try:
        atoms = read(args.structure)
    except Exception as e:
        print(f"Error reading structure {args.structure}: {e}")
        sys.exit(1)
        
    cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
    
    try:
        mag_sites = [int(idx.strip()) for idx in args.mag_sites.split(',')]
    except ValueError:
        print("Error: --mag-sites must be a comma-separated list of integers.")
        sys.exit(1)
        
    for idx in mag_sites:
        if idx < 0 or idx >= len(atoms):
            print(f"Error: Magnetic site index {idx} out of bounds (0 to {len(atoms)-1})")
            sys.exit(1)
            
    qpoint = args.qpoint
    
    print(f"Structure: {args.structure} ({len(atoms)} atoms)")
    print(f"Magnetic sites: {mag_sites}")
    print(f"q-point: {qpoint}")
    print("-" * 50)
    
    finder = MagneticTransitionFinder(cell, mag_sites, symprec=args.symprec)
    
    print("Computing magnetic irreps and basis vectors...")
    try:
        transitions = finder.find_transitions(qpoint)
    except Exception as e:
        print(f"Error computing transitions: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    if not transitions:
        print("No magnetic transitions found.")
        sys.exit(0)
        
    # Group by chiral vs achiral
    chiral_transitions = [t for t in transitions if t['is_chiral']]
    
    print("\n" + "=" * 80)
    print("CHIRAL MAGNETIC TRANSITIONS")
    print("=" * 80)
    if chiral_transitions:
        print(f"{'Irrep':>8} | {'Dim':>5} | {'OPD':>35} | {'Magnetic Space Group':>20}")
        print("-" * 80)
        for t in chiral_transitions:
            opd_str = str(np.round(t['opd'], 3))
            print(f"{t['irrep_index']:>8} | {t['irrep_dim']:>5} | {opd_str:>35} | {t['uni_number']} ({t['bns_number']})")
    else:
        print("No chiral magnetic transitions found at this q-point.")
        
    print("\n(Run with a different q-point or check all subgroups for more results)")

if __name__ == "__main__":
    main()
