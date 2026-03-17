"""Example: Label phonon modes from anaddb PHBST output.

This example demonstrates how to use symphon to analyze
phonon modes from ABINIT's anaddb output.

How to run:
    python label_example.py

Expected behavior:
    - Analyzes Gamma, X, and M points from LAO_PHBST.nc
    - Saves results to mode_labels.txt

Data:
    - Uses LAO_PHBST.nc as example input
    - PHBST file contains 172 q-points along Gamma-X-M path
"""

import io
import sys

import numpy as np
from symphon import print_irreps
from symphon.abipy_io import read_phbst_freqs_and_eigvecs

# Input PHBST file
phbst_fname = './LAO_PHBST.nc'

# Output file for results
output_fname = 'mode_labels.txt'

# Read available q-points from the PHBST file
atoms, qpoints, freqs, eig_vecs = read_phbst_freqs_and_eigvecs(phbst_fname)

print(f"Found {len(qpoints)} q-points in {phbst_fname}")

# Define q-points to analyze with their indices in the PHBST file
points_to_analyze = [
    {'name': 'Gamma', 'index': 0,  'kpname': None},
    {'name': 'X',     'index': 20, 'kpname': 'X'},
    {'name': 'M',     'index': 40, 'kpname': 'M'},
]

all_outputs = []

for point in points_to_analyze:
    print(f"\nAnalyzing {point['name']} point (index {point['index']})...")

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        print_irreps(
            phbst_fname=phbst_fname,
            ind_q=point['index'],
            symprec=1e-5,
            degeneracy_tolerance=1e-4,
            log_level=0,
            show_verbose=False,
            kpname=point['kpname'],
        )

        output = sys.stdout.getvalue()
        all_outputs.append(f"# {point['name']} point (q-point index {point['index']})\n{output}")
        print(f"  Success!")

    except Exception as e:
        print(f"  Error: {e}")
        all_outputs.append(f"# {point['name']} point (q-point index {point['index']})\n# Error: {e}\n")
    finally:
        sys.stdout = old_stdout

# Write all outputs to file
with open(output_fname, 'w') as f:
    for output in all_outputs:
        f.write(output)
        f.write("\n")

print(f"\nMode labeling complete. Results written to {output_fname}")
