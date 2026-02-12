"""Example: Compute irreps for BaTiO3 using phonopy params YAML.

How to run:
    python get_irreps_phonopy_BaTiO3.py

Expected behavior:
    Prints a table of phonon modes with their irreducible representations,
    frequencies, and IR/Raman activity at the Gamma point.
"""

from anaddb_irreps import print_irreps_phonopy


if __name__ == "__main__":
    # Path to the example phonopy params/YAML file
    params_path = "BaTiO3_phonopy_params.yaml"

    # Simple usage at Gamma (0, 0, 0)
    print_irreps_phonopy(params_path, qpoint=[0.0, 0.0, 0.0])
