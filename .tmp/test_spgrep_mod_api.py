import numpy as np
from spgrep_modulation.modulation import Modulation
from phonopy import load as phonopy_load
import os

# Create a minimal phonopy file or use existing one if possible
# Let's try to use the one from examples
phonopy_yaml = "examples/2_basic_phonopy/BaTiO3_phonopy_params.yaml"

if not os.path.exists(phonopy_yaml):
    # Fallback if not in the right dir
    phonopy_yaml = "/Users/hexu/projects/anaddb_irreps_dev/anaddb_irreps/examples/2_basic_phonopy/BaTiO3_phonopy_params.yaml"

ph = phonopy_load(phonopy_yaml)
qpoint = [0, 0, 0] # Gamma point has degenerate modes in cubic BaTiO3

md = Modulation.with_supercell_and_symmetry_search(
    dynamical_matrix=ph.dynamical_matrix,
    supercell_matrix=np.eye(3, dtype=int),
    qpoint=qpoint,
    factor=ph.unit_conversion_factor
)

print(f"Number of eigenspaces: {len(md.eigenspaces)}")
for i, (freq, eigvecs, irrep) in enumerate(md.eigenspaces):
    dim = eigvecs.shape[1]
    print(f"Eigenspace {i}: freq={freq:.2f}, dim={dim}")
    
    if dim > 1:
        print(f"  Testing alternative way for degenerate mode (dim={dim})")
        # Alternative 1: get_modulated_supercell_and_modulation with manual amplitudes
        amplitudes = [0.1] * dim
        arguments = [0.0] * dim
        try:
            cell, modulation = md.get_modulated_supercell_and_modulation(i, amplitudes, arguments)
            print(f"    Success: get_modulated_supercell_and_modulation returned cell and modulation of shape {modulation.shape}")
        except Exception as e:
            print(f"    Failed get_modulated_supercell_and_modulation: {e}")
            
        # Alternative 2: get_modulated_supercells
        try:
            cells = md.get_modulated_supercells(i, max_size=2)
            print(f"    Success: get_modulated_supercells returned {len(cells)} cells")
        except Exception as e:
            print(f"    Failed get_modulated_supercells: {e}")
            
        break
