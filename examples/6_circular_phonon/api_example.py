"""Example: Circular phonon analysis using the Python API.

How to run:
    cd 6_circular_phonon
    python api_example.py

Section 1 (SAM Calculator) runs live. Sections 2-3 show API usage patterns
as reference — use the CLI commands from cli_example.sh for those features.

In a clean install with all dependencies resolved, you can use:
    from symphon.chiral.sam import SAMCalculator
    from symphon.chiral.abstract_circular import AbstractCircularPhononFinder
"""

import importlib.util
import os
import numpy as np

_PKG = os.path.join(os.path.dirname(__file__), '..', '..', 'symphon', 'chiral')


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sam_mod = _load_module('sam', os.path.join(_PKG, 'sam.py'))
SAMCalculator = _sam_mod.SAMCalculator


def example_sam_calculator():
    """Demonstrate SAM (Spin Angular Momentum) calculation."""
    print("=" * 70)
    print("1. SAM Calculator (live)")
    print("=" * 70)
    print()
    print("The SAM is defined as (Zhang & Niu, PRL 112, 085503 (2014)):")
    print("  S = hbar * sum_kappa Im(epsilon*_kappa x epsilon_kappa)")
    print("  where epsilon is the mass-weighted eigenvector (phonopy convention).")
    print()

    linear_disp = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    sam_linear = SAMCalculator.calculate(linear_disp)
    circ_linear = SAMCalculator.get_circularity(sam_linear)
    print(f"Linear (real) displacement:")
    print(f"  SAM = {sam_linear}          (expected: [0, 0, 0])")
    print(f"  Circularity = {circ_linear:.6f}  (expected: 0)")
    print()

    circular_disp = np.zeros((3, 3), dtype=complex)
    circular_disp[0] = [1.0, 1j, 0.0]
    circular_disp[1] = [0.0, 0.0, 1.0]
    circular_disp[2] = [0.0, 0.0, 0.5 + 0.5j]

    sam_circular = SAMCalculator.calculate(circular_disp)
    circ_circular = SAMCalculator.get_circularity(sam_circular)
    print(f"Circular displacement (e_x + i*e_y):")
    print(f"  SAM = {sam_circular}")
    print(f"  Circularity = {circ_circular:.6f}  (expected: > 0)")
    print()

    masses = np.array([28.0, 16.0, 12.0])
    geo_circ = SAMCalculator.calculate_geometric_circularity(circular_disp, masses)
    print(f"Geometric circularity (mass-independent):")
    print(f"  S_geom = {geo_circ}")
    print()

    physical_disp = np.zeros((3, 3), dtype=complex)
    physical_disp[0] = [0.1, 0.05j, 0.0]
    physical_disp[1] = [0.0, 0.02, 0.01]
    physical_disp[2] = [0.03, 0.0, 0.0]
    sam_physical = SAMCalculator.calculate(physical_disp, masses=masses, mass_weighted=False)
    print(f"Physical (non-mass-weighted) displacement with masses {masses}:")
    print(f"  SAM = {sam_physical}")
    print()


def example_abstract_finder_reference():
    """Show API usage for group-theory circular phonon prediction."""
    print("=" * 70)
    print("2. Abstract Circular Phonon Finder (reference)")
    print("=" * 70)
    print()
    print("Identifies multidimensional irreducible representations at special")
    print("k-points — the necessary condition for circular polarization.")
    print()
    print("API usage:")
    print()
    print("  from symphon.chiral.abstract_circular import AbstractCircularPhononFinder")
    print()
    print("  finder = AbstractCircularPhononFinder(spg_number=198)  # P2_1_3")
    print("  candidates = finder.find_candidates()")
    print()
    print("  for kp in candidates:")
    print("      print(f\"k-point: {kp['kpname']}\")")
    print("      for c in kp['candidates']:")
    print("          print(f\"  IR: {c['name']}, dim={c['dim']}, OPDs: {c['possible_opds']}\")")
    print()
    print("  # Check Sohncke status:")
    print("  from symphon.chiral.sohncke import is_sohncke, get_sohncke_class")
    print("  is_sohncke(198)        # True")
    print("  get_sohncke_class(198) # SohnckeClass.CLASS_III")
    print()
    print("CLI equivalent:")
    print("  symphon circular-abstract --sg 198")
    print("  symphon circular-abstract --all-sohncke")
    print()


def example_concrete_finder_reference():
    """Show API usage for full circular phonon analysis from PHBST."""
    print("=" * 70)
    print("3. Concrete Finder (from PHBST file) — reference")
    print("=" * 70)
    print()
    print("API usage:")
    print()
    print("  from symphon.chiral import CircularPhononFinder")
    print("  from symphon.io.phbst import read_phbst_freqs_and_eigvecs, ase_atoms_to_phonopy_atoms")
    print()
    print("  atoms, qpoints, freqs, evecs = read_phbst_freqs_and_eigvecs('run_PHBST.nc')")
    print("  primitive = ase_atoms_to_phonopy_atoms(atoms)")
    print("  finder = CircularPhononFinder(")
    print("      primitive=primitive, qpoint=qpoints[0],")
    print("      freqs=freqs[0], eigvecs=evecs[0],")
    print("  )")
    print("  results = finder.run()")
    print()
    print("  for mode in results:")
    print("      print(f'Band {mode.band_index}: f={mode.frequency:.3f} THz, '")
    print("            f'circ={mode.circularity:.3f}, hand={mode.handedness}, '")
    print("            f'irrep={mode.irrep_label}, opd={mode.opd}')")
    print()
    print("CLI equivalent:")
    print("  symphon-circular run_PHBST.nc")
    print("  symphon-circular run_PHBST.nc --qpoint 0.5 0.0 0.0")
    print()


if __name__ == "__main__":
    example_sam_calculator()
    example_abstract_finder_reference()
    example_concrete_finder_reference()

    print("=" * 70)
    print("Done! See cli_example.sh for runnable CLI commands.")
    print("=" * 70)
