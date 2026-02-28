#!/usr/bin/env python
"""
CLI tool for finding abstract chiral magnetic phase transitions.

Usage:
    abstract-magnetic-chiral --spg 221 --qpoint 0 0 0
"""

import argparse
import sys
import numpy as np
from anaddb_irreps.abstract_magnetic import AbstractMagneticTransitionFinder

def parse_args():
    parser = argparse.ArgumentParser(description="Find abstract chiral magnetic transitions")
    parser.add_argument("--spg", type=int, required=True, help="Parent space group number (1-230)")
    parser.add_argument("--qpoint", nargs=3, type=float, default=[0, 0, 0], help="Propagation wavevector (fractional)")
    parser.add_argument("--multi-k", action="store_true", help="Include multi-k transitions")
    parser.add_argument("--all", action="store_true", help="Show all transitions, not just chiral ones")
    parser.add_argument("--symprec", type=float, default=1e-5, help="Symmetry precision")
    return parser.parse_args()

def main():
    args = parse_args()
    spg = args.spg
    qpoint = args.qpoint
    
    if spg < 1 or spg > 230:
        print("Error: Space group number must be between 1 and 230.")
        sys.exit(1)
        
    print(f"Parent Space Group: {spg}")
    print(f"q-point: {qpoint}")
    print("-" * 50)
    
    finder = AbstractMagneticTransitionFinder(spg, symprec=args.symprec)
    
    print("Computing abstract magnetic irreps and subgroups...")
    try:
        transitions = finder.find_transitions(qpoint, include_multi_k=args.multi_k)
    except Exception as e:
        print(f"Error computing transitions: {e}")
        sys.exit(1)
        
    if not transitions:
        print("No magnetic transitions found.")
        sys.exit(0)
        
    if not args.all:
        filtered_transitions = [t for t in transitions if t['is_chiral']]
        title = "CHIRAL MAGNETIC TRANSITIONS"
    else:
        filtered_transitions = transitions
        title = "ALL MAGNETIC TRANSITIONS"
        
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if filtered_transitions:
        print(f"{'Irrep':>8} | {'Dim':>5} | {'k-type':>8} | {'Magnetic Space Group':>20} | {'Chiral':>8}")
        print("-" * 80)
        for t in filtered_transitions:
            chiral_str = "Yes" if t['is_chiral'] else "No"
            print(f"{t['irrep_index']:>8} | {t['irrep_dim']:>5} | {t.get('k_type', '1-k'):>8} | {t['uni_number']:>6} ({t['bns_number']:>11}) | {chiral_str:>8}")
    else:
        print("No matches found.")

if __name__ == "__main__":
    main()
