# anaddb_irreps

A simple wrapper of the phonopy irreps module for finding irreducible representations of phonon modes in ABINIT's anaddb output.

## Features

- **Automatic high-symmetry analysis**: Analyzes all high-symmetry k-points (GM, X, M, R, etc.) automatically
- **Dual-label display at Gamma**: Shows both Mulliken (T1u, Eg, etc.) and BCS (GM4-, GM5+, etc.) notation side-by-side
- **Spectroscopic activity**: Identifies IR- and Raman-active modes at Γ point
- **Multiple input formats**: Supports both anaddb PHBST NetCDF files and phonopy params/YAML files
- **Dual backend support**:
  - `phonopy` backend: Γ-point analysis with IR/Raman activity (used automatically at Gamma)
  - `irrep` backend: Non-Gamma point analysis using the `irrep` package (used automatically for other points)
- **CLI and Python API**: Flexible usage through command-line tools or Python functions
- **Complete symmetry information**: Reports both space group and point group for the crystal

## Installation

### Basic Installation

Install from PyPI:
```bash
pip install anaddb_irreps
```

### Optional Dependencies

For **non-Gamma phonons** (k-points other than Γ), install with the `irrep` backend:
```bash
pip install "anaddb_irreps[irrep]"
```

For **ABINIT support** (reading PHBST NetCDF files), install with the `abipy` optional dependency:
```bash
pip install "anaddb_irreps[abipy]"
```

To install all optional dependencies:
```bash
pip install "anaddb_irreps[irrep,abipy]"
```

## Documentation

- **[Usage Guide](docs/usage.md)**: Complete API and CLI reference with examples for all features
- **[Irreducible Representation Labeling Guide](docs/irreps_guide.md)**: Mathematical foundations, Mulliken notation, and tutorial for non-Gamma phonons (e.g., X and M points in BaTiO3)
- **[IR and Raman Activity Methodology](docs/activity_methodology.md)**: Scientific guide to the group-theoretical method used to identify spectroscopic activity, with TmFeO3 case study

## Quick Start

### Python API

```python
from anaddb_irreps import print_irreps_phonopy

# Analyze Gamma point
print_irreps_phonopy("phonopy_params.yaml", qpoint=[0, 0, 0])

# Show both Mulliken and BCS labels at Gamma
print_irreps_phonopy("phonopy_params.yaml", qpoint=[0, 0, 0], both_labels=True)

# Analyze non-Gamma point (requires irrep backend)
print_irreps_phonopy(
    "phonopy_params.yaml",
    qpoint=[0.5, 0.5, 0],
    backend="irrep",
    kpname="M"
)
```

### Command Line

#### phonopy-irreps (Auto-discovery mode)

```bash
# Analyze all high-symmetry k-points automatically
phonopy-irreps --params phonopy_params.yaml

# Output shows all high-symmetry points:
# - GM point: dual labels (Mulliken + BCS) + IR/Raman activity
# - Other points (M, R, X, etc.): BCS labels
```

#### anaddb-irreps (Manual mode for PHBST files)

```bash
# PHBST files require explicit q-point index (no auto-discovery)
anaddb-irreps --phbst run_PHBST.nc --q-index 0

# For non-Gamma points, specify backend and k-point name
anaddb-irreps --phbst run_PHBST.nc --q-index 5 --backend irrep --kpname M
```

## Output Examples

### Automatic high-symmetry analysis (phonopy-irreps)

When you run `phonopy-irreps --params phonopy_params.yaml`, it automatically:
1. Discovers all high-symmetry k-points from the space group
2. Shows dual labels (Mulliken + BCS) at Gamma point with IR/Raman activity
3. Shows BCS labels for other high-symmetry points

```
Space group: Pm-3m
Found 4 high-symmetry points:
  GM: k=(0.0, 0.0, 0.0)
  M: k=(0.5, 0.5, 0.0)
  R: k=(0.5, 0.5, 0.5)
  X: k=(0.0, 0.5, 0.0)

# GM point (k=(0.0, 0.0, 0.0))
============================================================
q-point: [0.0000, 0.0000, 0.0000]
Space group: Pm-3m
Point group: m-3m

# qx      qy      qz      band  freq(THz)   freq(cm-1)   label(M)   label(BCS)  IR  Raman
 0.0000  0.0000  0.0000     0     -6.0487      -201.76  T1u         GM4-         Y    .  
 0.0000  0.0000  0.0000     9      8.5335       284.65  T2u         GM5-         .    .  

# X point (k=(0.0, 0.5, 0.0))
============================================================
q-point: [0.0000, 0.5000, 0.0000]
Space group: Pm-3m
Point group: m-3m

# qx      qy      qz      band  freq(THz)   freq(cm-1)   label
 0.0000  0.5000  0.0000     0     -4.8804      -162.79  X5+
 0.0000  0.5000  0.0000     2      3.3171       110.65  X5-
```

## Examples

See `examples/MoS2_1T/` for a complete working example with MoS2 1T structure using anaddb output.

See `docs/irreps_guide.md` for BaTiO3 examples demonstrating both Gamma and non-Gamma point analysis.

## License

BSD-2-clause
