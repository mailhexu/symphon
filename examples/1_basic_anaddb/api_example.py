"""Example: Compute irreps for MoS2 1T structure using Python API.

How to run:
    python get_irreps.py

Expected behavior:
    Prints a table of phonon modes with their irreducible representations,
    frequencies, and IR/Raman activity.
"""

from anaddb_irreps import print_irreps

if __name__ == "__main__":
    # Simple usage with defaults
    print_irreps("run_PHBST.nc", ind_q=0)
    
    # With optional parameters:
    # print_irreps(
    #     "run_PHBST.nc",
    #     ind_q=0,
    #     symprec=1e-8,              # Symmetry precision (default: 1e-5)
    #     degeneracy_tolerance=1e-4, # Frequency tolerance (default: 1e-4)
    #     log_level=1,               # Verbosity: 0=quiet, 1+=more verbose (default: 0)
    #     show_verbose=True          # Show detailed phonopy output (default: False)
    # )
