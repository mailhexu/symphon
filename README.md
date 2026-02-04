# anaddb_irreps

A simple wrapper of the phonopy irreps module for finding irreducible representations of phonon modes in ABINIT's anaddb output.

## Installation

Install from PyPI:
```bash
pip install anaddb_irreps
```

To enable support for **non-Gamma phonons**, install with the `irrep` backend:
```bash
pip install "anaddb_irreps[irrep]"
```

## Documentation

For a detailed explanation of the mathematics, notation (Mulliken labels), and a tutorial for labeling non-Gamma phonons (like the X and M points in BaTiO3), see the **[Irreducible Representation Labeling Guide](docs/irreps_guide.md)**.

## Usage (Python API)

### From anaddb PHBST (AbiPy)

Run anaddb to get the PHBST file with phonon frequencies and eigenvectors. See example in `examples/MoS2_1T/anaddb_input/`.

#### Basic Usage

```python
from anaddb_irreps import print_irreps

# Simple usage
print_irreps("run_PHBST.nc", ind_q=0)
```

### With Options

```python
from anaddb_irreps import print_irreps

print_irreps(
    "run_PHBST.nc",
    ind_q=0,
    symprec=1e-8,              # Symmetry precision (default: 1e-5)
    degeneracy_tolerance=1e-4, # Frequency tolerance (default: 1e-4)
    log_level=1,               # Verbosity: 0=quiet, 1+=verbose (default: 0)
    show_verbose=True          # Show detailed phonopy output (default: False)
)
```

### From phonopy params/YAML

If you already have a phonopy params/YAML file (e.g. `phonopy_params.yaml` or
`phonopy.yaml`), you can use the phonopy-based helper:

```python
from anaddb_irreps import print_irreps_phonopy

irr = print_irreps_phonopy(
    "phonopy_params.yaml",
    qpoint=[0.0, 0.0, 0.0],
    symprec=1e-5,
    degeneracy_tolerance=1e-4,
    log_level=0,
    show_verbose=False,
)
```

### Parameters

For `print_irreps` (anaddb route):

- **phbst_fname** (str, required): Path to PHBST NetCDF file
- **ind_q** (int, required): Index of q-point in PHBST file (0-based)
- **symprec** (float): Symmetry precision for structure analysis (default: 1e-5)
- **degeneracy_tolerance** (float): Frequency tolerance for degeneracy detection (default: 1e-4)
- **is_little_cogroup** (bool): Use little co-group setting (default: False)
- **log_level** (int): Verbosity level; 0=quiet, higher=more verbose (default: 0)
- **show_verbose** (bool): Show detailed phonopy irreps output (default: False)

For `print_irreps_phonopy` (phonopy route):

- **phonopy_params** (str, required): Path to phonopy params/YAML file
- **qpoint** (sequence of 3 floats, required): q-point in fractional coordinates
- **symprec** (float or None): Symmetry precision for structure analysis. If `None` (or omitted), anaddb_irreps will try to use the symmetry tolerance recorded in the phonopy file (e.g. `phonopy.symmetry_tolerance` in the YAML), falling back to `1e-5` when not available.
- **degeneracy_tolerance** (float): Frequency tolerance for degeneracy detection (default: 1e-4)
- **is_little_cogroup** (bool): Use little co-group setting (default: False)
- **log_level** (int): Verbosity level; 0=quiet, higher=more verbose (default: 0)
- **show_verbose** (bool): Show detailed phonopy irreps output (default: False)
 
## Usage (CLI)

Use the `anaddb-irreps` and `phonopy-irreps` command-line tools for quick command-line access.

### Basic Example (anaddb PHBST)

```bash
anaddb-irreps --phbst run_PHBST.nc --q-index 0
```

### With Options (anaddb PHBST)

```bash
anaddb-irreps \
  --phbst run_PHBST.nc \
  --q-index 0 \
  --symprec 1e-8 \
  --degeneracy-tolerance 1e-4 \
  --log-level 1
```

### Basic Example (phonopy params)

```bash
phonopy-irreps \
  --params phonopy_params.yaml \
  --qpoint 0.0 0.0 0.0
```

### CLI Options

For `anaddb-irreps`:

- `-p`, `--phbst` (required): Path to PHBST NetCDF file
- `-q`, `--q-index` (required): Index of q-point in PHBST file (0-based)
- `-s`, `--symprec`: Symmetry precision (default: 1e-5)
- `-d`, `--degeneracy-tolerance`: Frequency tolerance for degeneracy (default: 1e-4)
- `-l`, `--is-little-cogroup`: Use little co-group setting
- `-v`, `--log-level`: Verbosity level; 0=quiet, higher=more verbose (default: 0)

For `phonopy-irreps`:

- `-p`, `--params` (required): Path to phonopy params/YAML file
- `--qpoint` (required): Three floats for q-point in fractional coordinates
- `-s`, `--symprec`: Override symmetry precision. If omitted, anaddb_irreps will try to use the symmetry tolerance stored in the phonopy file, falling back to `1e-5`.
- `-d`, `--degeneracy-tolerance`: Frequency tolerance for degeneracy (default: 1e-4)
- `-l`, `--is-little-cogroup`: Use little co-group setting
- `-v`, `--log-level`: Verbosity level; 0=quiet, higher=more verbose (default: 0)

The CLI produces the same output format as the Python API, showing q-point, point group, and a table of phonon modes with their irreducible representations and IR/Raman activity.


## Output Format

The output includes:

1. **Q-point coordinates** in fractional coordinates
2. **Point group** of the structure at that q-point
3. **Mode table** with columns:
   - `qx, qy, qz`: q-point coordinates (repeated for each mode)
   - `band`: Mode index (0-based)
   - `freq(THz)`: Frequency in THz
   - `freq(cm-1)`: Frequency in cm⁻¹
   - `label`: Irreducible representation label (e.g., Eu, Eg, A1g)
   - `IR`: IR activity (`Y` = active, `.` = inactive)
   - `Raman`: Raman activity (`Y` = active, `.` = inactive)

### Example Output

```
q-point: [0.0000, 0.0000, 0.0000]
Point group: -3m

# qx      qy      qz      band  freq(THz)   freq(cm-1)   label        IR  Raman
 0.0000  0.0000  0.0000     0      0.0000         0.00  -            .    .  
 0.0000  0.0000  0.0000     1      0.0000         0.00  -            .    .  
 0.0000  0.0000  0.0000     2      0.0000         0.00  -            .    .  
 0.0000  0.0000  0.0000     3      6.4001       213.49  Eu           Y    .  
 0.0000  0.0000  0.0000     4      6.4001       213.49  Eu           Y    .  
 0.0000  0.0000  0.0000     5      6.8617       228.88  Eg           .    Y  
 0.0000  0.0000  0.0000     6      6.8617       228.88  Eg           .    Y  
 0.0000  0.0000  0.0000     7     11.1626       372.34  A1g          .    Y  
 0.0000  0.0000  0.0000     8     11.3152       377.43  A2u          Y    .  
```

## Examples

See `examples/MoS2_1T/` for a complete working example with MoS2 1T structure.

