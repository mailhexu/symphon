"""Example: Label phonon modes from anaddb PHBST output.

This example demonstrates how to use anaddb_irreps to analyze
phonon modes from ABINIT's anaddb output.

How to run:
    python label_example.py

Expected behavior:
    - Analyzes Gamma point from LAO_PHBST.nc
    - Prints irreducible representations and IR/Raman activity
    - Saves results to mode_labels.txt

Data:
    - Uses LAO_PHBST.nc as example input
    - Demonstrates phonopy backend for Gamma point analysis
    - PHBST file contains 172 q-points along Gamma-X path
"""

import numpy as np
from anaddb_irreps import print_irreps

# Input PHBST file
phbst_fname = './LAO_PHBST.nc'

# Output file for results
output_fname = 'mode_labels.txt'

# Read available q-points from the PHBST file
from anaddb_irreps.abipy_io import read_phbst_freqs_and_eigvecs
atoms, qpoints, freqs, eig_vecs = read_phbst_freqs_and_eigvecs(phbst_fname)

print(f"Found {len(qpoints)} q-points in {phbst_fname}")
print(f"Analyzing Gamma point (index 0)...")

# Analyze Gamma point (index 0)
ind_q = 0
backend = 'phonopy'

# Capture output
import io
import sys
old_stdout = sys.stdout
sys.stdout = io.StringIO()

print_irreps(
    phbst_fname=phbst_fname,
    ind_q=ind_q,
    symprec=1e-5,
    degeneracy_tolerance=1e-4,
    log_level=0,
    show_verbose=False,
    backend=backend
)

output = sys.stdout.getvalue()
sys.stdout = old_stdout

# Write to output file
with open(output_fname, 'w') as f:
    f.write(f"# Gamma point (q-point index {ind_q})\n")
    f.write(output)

print(f"\nMode labeling complete. Results written to {output_fname}")
