#!/usr/bin/env python
"""CLI tool for generating modulated supercells."""

import argparse
import sys
import numpy as np
from symphon.irreps_anaddb import IrRepsAnaddb, IrRepsPhonopy

def parse_args():
    parser = argparse.ArgumentParser(description="Generate modulated supercells using high-symmetry basis")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--phbst", help="Path to PHBST NetCDF file")
    group.add_argument("--params", help="Path to phonopy params/YAML file")
    
    parser.add_argument("--q-index", type=int, default=0, help="q-point index (for PHBST)")
    parser.add_argument("--qpoint", nargs=3, type=float, help="q-point coordinates (for Phonopy)")
    
    parser.add_argument("--mode", type=int, required=True, help="Mode index (0 to 3N-1)")
    parser.add_argument("--amplitude", type=float, default=0.1, help="Maximum displacement (Angstrom)")
    parser.add_argument("--supercell", nargs="+", type=int, help="Supercell matrix (3 or 9 integers)")
    
    parser.add_argument("--output", "-o", default="MPOSCAR", help="Output filename")
    parser.add_argument("--symprec", type=float, default=1e-5, help="Symmetry precision")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.phbst:
        irr = IrRepsAnaddb(
            phbst_fname=args.phbst,
            ind_q=args.q_index,
            symprec=args.symprec,
        )
    else:
        if args.qpoint is None:
            print("Error: --qpoint is required when using --params")
            sys.exit(1)
        irr = IrRepsPhonopy(
            phonopy_params=args.params,
            qpoint=args.qpoint,
            symprec=args.symprec,
        )
        
    irr.run()
    
    supercell_matrix = None
    if args.supercell:
        if len(args.supercell) == 3:
            supercell_matrix = np.diag(args.supercell)
        elif len(args.supercell) == 9:
            supercell_matrix = np.array(args.supercell).reshape(3, 3)
        else:
            print("Error: --supercell must have 3 or 9 integers")
            sys.exit(1)
            
    try:
        sc = irr.get_modulated_supercell(
            mode_index=args.mode,
            amplitude=args.amplitude,
            supercell_matrix=supercell_matrix
        )
        
        from phonopy.interface.vasp import write_vasp
        write_vasp(args.output, sc)
        print(f"Modulated supercell saved to {args.output}")
        
    except Exception as e:
        print(f"Error generating modulated supercell: {e}")
        # sys.exit(1)

if __name__ == "__main__":
    main()
