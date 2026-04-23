#!/usr/bin/env python
"""
CLI tool for finding circularly polarized phonons.

Usage:
    symphon-circular <PHBST.nc> [options]
"""

import argparse
import sys
import numpy as np
from symphon.chiral import CircularPhononFinder
from symphon.io.phbst import read_phbst_freqs_and_eigvecs, ase_atoms_to_phonopy_atoms

def main():
    parser = argparse.ArgumentParser(
        description="Find circularly polarized phonons from ABINIT phonon results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "phbst",
        help="Path to ABINIT _PHBST.nc file",
    )
    
    parser.add_argument(
        "-q", "--qpoint",
        nargs=3,
        type=float,
        default=None,
        help="q-point to analyze (default: 0 0 0 or first in file)",
    )
    
    parser.add_argument(
        "--symprec",
        type=float,
        default=1e-5,
        help="Symmetry precision (default: 1e-5)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity",
    )

    args = parser.parse_args()

    try:
        atoms, qpoints, freqs, evecs = read_phbst_freqs_and_eigvecs(args.phbst)
    except Exception as e:
        print(f"Error reading {args.phbst}: {e}", file=sys.stderr)
        sys.exit(1)

    # Find the requested q-point index
    if args.qpoint is not None:
        target_q = np.array(args.qpoint)
        found_idx = -1
        for i, q in enumerate(qpoints):
            diff = q - target_q
            if np.allclose(diff - np.round(diff), 0, atol=args.symprec):
                found_idx = i
                break
        if found_idx == -1:
            print(f"Error: q-point {target_q} not found in {args.phbst}", file=sys.stderr)
            sys.exit(1)
        q_idx = found_idx
    else:
        q_idx = 0
        target_q = qpoints[0]

    print(f"Analyzing q-point: {target_q}")
    primitive = ase_atoms_to_phonopy_atoms(atoms)
    
    finder = CircularPhononFinder(
        primitive=primitive,
        qpoint=target_q,
        freqs=freqs[q_idx],
        eigvecs=evecs[q_idx],
        symprec=args.symprec,
        log_level=args.verbose
    )
    
    results = finder.run()
    
    if not results:
        print("No modes found at this q-point.")
        return

    print(f"Found {len(results)} modes (including linear bases and circular combinations):")
    print("-" * 110)
    print(f"{'Band':>5} {'Freq(THz)':>10} {'Freq(cm-1)':>10} {'Irrep':>10} {'Dim':>3} {'Circ':>6} {'Hand':>4} {'OPD':>12} {'SAM Vector':>30}")
    print("-" * 110)
    
    for res in results:
        sam_str = f"[{res.sam[0]:.3f}, {res.sam[1]:.3f}, {res.sam[2]:.3f}]"
        opd_str = res.opd if res.opd else "N/A"
        print(f"{res.band_index:>5} {res.frequency:>10.4f} {res.frequency_cm1:>10.4f} {res.irrep_label:>10} {res.dimension:>3} {res.circularity:>6.3f} {res.handedness:>4} {opd_str:>12} {sam_str:>30}")
    print("-" * 110)

if __name__ == "__main__":
    main()
