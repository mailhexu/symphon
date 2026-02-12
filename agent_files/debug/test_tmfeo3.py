#!/usr/bin/env python3
"""
Test script for TmFeO3 phonopy irrep analysis.

This script tests the phonopy-irreps CLI with TmFeO3 (space group Pnma, #62).
Tests auto-discovery of high-symmetry points with dual labels at Gamma.

How to run:
    uv run python agent_files/debug/test_tmfeo3.py

What it tests:
    - Auto-discovery of high-symmetry k-points for Pnma (62)
    - Dual labels (Mulliken + BCS) at Gamma point
    - Clean output format (no redundant qx/qy/qz columns)
    - IR/Raman activity at Gamma
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run phonopy-irreps on TmFeO3 example."""
    # Path to TmFeO3 params file
    params_file = Path("examples/3_advanced_phonopy/TmFeO3_phonopy_params.yaml")
    
    if not params_file.exists():
        print(f"Error: {params_file} not found")
        sys.exit(1)
    
    print("=" * 70)
    print("Testing phonopy-irreps with TmFeO3 (Pnma, space group 62)")
    print("=" * 70)
    print()
    
    # Run phonopy-irreps
    cmd = ["uv", "run", "phonopy-irreps", "--params", str(params_file)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
        
        print()
        print("=" * 70)
        print("Test completed successfully!")
        print("=" * 70)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running phonopy-irreps: {e}", file=sys.stderr)
        if e.stdout:
            print("stdout:", file=sys.stderr)
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print("stderr:", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
