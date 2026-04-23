#!/usr/bin/env python
"""
CLI tool for identifying potential circular phonon IRs from space group symmetry.

Usage:
    symphon circular-abstract <spg_num> [options]
"""

import argparse
import sys
import numpy as np
from symphon.chiral.abstract_circular import AbstractCircularPhononFinder

def main():
    parser = argparse.ArgumentParser(
        description="Identify potential circular phonon IRs purely from space group symmetry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sg", "--space-group",
        type=int,
        help="Space group number (1-230)",
    )
    group.add_argument(
        "--all-sohncke",
        action="store_true",
        help="Show candidates for all 65 Sohncke space groups",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show coordinate k-points",
    )

    args = parser.parse_args()

    if args.all_sohncke:
        from symphon.chiral.abstract_circular import get_all_sohncke_candidates
        all_results = get_all_sohncke_candidates()
        print(f"{'SG':>4} {'Symbol':>12} {'k-points':<10} {'Multidim IRs':<20}")
        print("-" * 60)
        from symphon.chiral.sohncke import is_sohncke
        from irreptables.irreps import IrrepTable
        for sg_num in sorted(all_results.keys()):
            table = IrrepTable(str(sg_num), spinor=False)
            results = all_results[sg_num]
            kp_str = ", ".join(res['kpname'] for res in results)
            irrep_count = sum(len(res['candidates']) for res in results)
            print(f"{sg_num:>4} {table.name:>12} {kp_str:<10} {irrep_count:<20}")
        return

    if args.sg:
        if not 1 <= args.sg <= 230:
            print(f"Error: Invalid space group {args.sg}")
            sys.exit(1)
            
        finder = AbstractCircularPhononFinder(args.sg)
        results = finder.find_candidates()
        
        if not results:
            print(f"No multidimensional IRs found for Space Group {args.sg}.")
            return

        print(f"Candidate IRs for Circular Phonons in Space Group {args.sg} ({finder.table.name}):")
        print("-" * 60)
        print(f"{'k-point':<10} {'Coords':<20} {'IR Name':<12} {'Dim':<5} {'Possible OPDs'}")
        print("-" * 60)
        
        for res in sorted(results, key=lambda x: x['kpname']):
            kpname = res['kpname']
            coords = f"[{res['kcoords'][0]:.2f}, {res['kcoords'][1]:.2f}, {res['kcoords'][2]:.2f}]"
            for cand in res['candidates']:
                opds = ", ".join(cand['possible_opds'])
                print(f"{kpname:<10} {coords:<20} {cand['name']:<12} {cand['dim']:<5} {opds}")
        print("-" * 60)

if __name__ == "__main__":
    main()
