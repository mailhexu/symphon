# anaddb_irreps

A simple wrapper of the phonopy irreps module for finding irreducible representations of phonon modes in ABINIT's anaddb output.

## Features

- **Multiple input formats**: Supports both anaddb PHBST NetCDF files and phonopy params/YAML files
- **Dual backend support**:
  - `phonopy` backend: Standard Γ-point analysis with IR/Raman activity identification
  - `irrep` backend: Non-Gamma point analysis using the `irrep` package
- **CLI and Python API**: Flexible usage through command-line tools or Python functions
- **Complete symmetry information**: Reports both space group and point group for the crystal
- **Spectroscopic activity**: Identifies IR- and Raman-active modes (at Γ point)

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

# Analyze non-Gamma point (requires irrep backend)
print_irreps_phonopy(
    "phonopy_params.yaml",
    qpoint=[0.5, 0.5, 0],
    backend="irrep",
    kpname="M"
)
```

### Command Line

```bash
# Analyze Gamma point
phonopy-irreps --params phonopy_params.yaml --qpoint 0 0 0

# Analyze non-Gamma point
phonopy-irreps --params phonopy_params.yaml --qpoint 0 0.5 0 --backend irrep --kpname X
```

## Output

```
q-point: [0.0000, 0.5000, 0.0000]
Space group: Pm-3m
Point group: m-3m

# qx      qy      qz      band  freq(THz)   freq(cm-1)   label
  0.0000  0.5000  0.0000     0     -4.8804      -162.79  X5+
  0.0000  0.5000  0.0000     1     -4.8804      -162.79  X5+
  0.0000  0.5000  0.0000     2      3.3171       110.65  X5-
```

## Examples

See `examples/MoS2_1T/` for a complete working example with MoS2 1T structure using anaddb output.

See `docs/irreps_guide.md` for BaTiO3 examples demonstrating both Gamma and non-Gamma point analysis.

## License

BSD-2-clause
