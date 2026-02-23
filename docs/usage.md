# Usage Guide

This guide provides detailed usage instructions for `anaddb_irreps` including both Python API and command-line interfaces.

## Table of Contents

- [Python API](#python-api)
  - [From anaddb PHBST (AbiPy)](#from-anaddb-phbst-abipy)
  - [From phonopy params/YAML](#from-phonopy-paramsyaml)
  - [Parameters](#parameters)
- [Command-Line Interface](#command-line-interface)
  - [anaddb-irreps](#anaddb-irreps)
  - [phonopy-irreps](#phonopy-irreps)
  - [CLI Options](#cli-options)
- [Output Format](#output-format)
  - [Example Output](#example-output)
- [Chiral Phase Transitions](#chiral-phase-transitions)
  - [Additional Columns](#additional-columns)
  - [Example: BaTiO₃ at Gamma](#example-batio₃-at-gamma)
  - [Example: TmFeO₃ at Gamma](#example-tmfeo₃-at-gamma)
  - [Example: Class II Enantiomorphous Pairs](#example-class-ii-enantiomorphous-pairs)
  - [When Chiral Transitions Appear](#when-chiral-transitions-appear)
  - [Interpreting OPD Notation](#interpreting-opd-notation)

---

## Python API

### From anaddb PHBST (AbiPy)

Run anaddb to get the PHBST file with phonon frequencies and eigenvectors. See example in `examples/MoS2_1T/anaddb_input/`.

#### Basic Usage

```python
from anaddb_irreps import print_irreps

# Simple usage
print_irreps("run_PHBST.nc", ind_q=0)
```

#### With Options

```python
from anaddb_irreps import print_irreps

print_irreps(
    "run_PHBST.nc",
    ind_q=0,
    symprec=1e-8,              # Symmetry precision (default: 1e-5)
    degeneracy_tolerance=1e-4, # Frequency tolerance (default: 1e-4)
    is_little_cogroup=False,    # Use little co-group setting (default: False)
    log_level=0,               # Verbosity: 0=quiet, 1+=verbose (default: 0)
    show_verbose=True,          # Show detailed phonopy output (default: False)
    backend="phonopy",         # Backend: "phonopy" or "irrep" (default: "phonopy")
    kpname=None,               # K-point name for irrep backend (e.g., "GM", "X", "M")
)
```

### From phonopy params/YAML

If you already have a phonopy params/YAML file (e.g. `phonopy_params.yaml` or `phonopy.yaml`), you can use the phonopy-based helper:

#### Basic Usage

```python
from anaddb_irreps import print_irreps_phonopy

# Simple usage at Gamma point
print_irreps_phonopy("phonopy_params.yaml", qpoint=[0.0, 0.0, 0.0])
```

#### With Options

```python
from anaddb_irreps import print_irreps_phonopy

# For non-Gamma points using irrep backend
print_irreps_phonopy(
    "phonopy_params.yaml",
    qpoint=[0.5, 0.5, 0.0],  # M point
    symprec=1e-5,
    degeneracy_tolerance=1e-4,
    is_little_cogroup=False,
    log_level=0,
    show_verbose=False,
    backend="irrep",              # Backend: "phonopy" or "irrep"
    kpname="M",                  # K-point name (required for non-Gamma with irrep backend)
)
```

### Parameters

#### For `print_irreps` (anaddb route):

- **phbst_fname** (str, required): Path to PHBST NetCDF file
- **ind_q** (int, required): Index of q-point in PHBST file (0-based)
- **symprec** (float): Symmetry precision for structure analysis (default: 1e-5)
- **degeneracy_tolerance** (float): Frequency tolerance for degeneracy detection (default: 1e-4)
- **is_little_cogroup** (bool): Use little co-group setting (default: False)
- **log_level** (int): Verbosity level; 0=quiet, higher=more verbose (default: 0)
- **show_verbose** (bool): Show detailed phonopy irreps output (default: False)
- **backend** (str): Backend driver: `"phonopy"` or `"irrep"` (default: `"phonopy"`)
- **kpname** (str): K-point name for `irrep` backend (e.g., `"GM"`, `"X"`, `"M"`). Required for non-Gamma points when using irrep backend.

#### For `print_irreps_phonopy` (phonopy route):

- **phonopy_params** (str, required): Path to phonopy params/YAML file
- **qpoint** (sequence of 3 floats, required): q-point in fractional coordinates
- **symprec** (float or None): Symmetry precision for structure analysis. If `None` (or omitted), anaddb_irreps will try to use the symmetry tolerance recorded in the phonopy file (e.g., `phonopy.symmetry_tolerance` in the YAML), falling back to `1e-5` when not available.
- **degeneracy_tolerance** (float): Frequency tolerance for degeneracy detection (default: 1e-4)
- **is_little_cogroup** (bool): Use little co-group setting (default: False)
- **log_level** (int): Verbosity level; 0=quiet, higher=more verbose (default: 0)
- **show_verbose** (bool): Show detailed phonopy irreps output (default: False)
- **backend** (str): Backend driver: `"phonopy"` or `"irrep"` (default: `"phonopy"`)
- **kpname** (str): K-point name for `irrep` backend (e.g., `"GM"`, `"X"`, `"M"`). Required for non-Gamma points when using irrep backend.

---

## Command-Line Interface

Use the `anaddb-irreps` and `phonopy-irreps` command-line tools for quick command-line access.

### anaddb-irreps

For analyzing anaddb PHBST NetCDF files.

#### Basic Example

```bash
anaddb-irreps --phbst run_PHBST.nc --q-index 0
```

#### With Options

```bash
anaddb-irreps \
  --phbst run_PHBST.nc \
  --q-index 0 \
  --symprec 1e-8 \
  --degeneracy-tolerance 1e-4 \
  --log-level 1
```

#### Using irrep Backend

```bash
# For non-Gamma points, use irrep backend
anaddb-irreps \
  --phbst run_PHBST.nc \
  --q-index 1 \
  --backend irrep \
  --kpname X
```

### phonopy-irreps

For analyzing phonopy params/YAML files.

#### Basic Example

```bash
phonopy-irreps --params phonopy_params.yaml --qpoint 0.0 0.0 0.0
```

#### With Options

```bash
phonopy-irreps \
  --params phonopy_params.yaml \
  --qpoint 0.5 0.5 0.0 \
  --symprec 1e-5 \
  --degeneracy-tolerance 1e-4 \
  --log-level 0
```

#### Using irrep Backend

```bash
# For non-Gamma points, use irrep backend
phonopy-irreps \
  --params phonopy_params.yaml \
  --qpoint 0.0 0.5 0.0 \
  --backend irrep \
  --kpname X
```

### CLI Options

#### For `anaddb-irreps`:

- `-p`, `--phbst` (required): Path to PHBST NetCDF file
- `-q`, `--q-index` (required): Index of q-point in PHBST file (0-based)
- `-s`, `--symprec`: Symmetry precision (default: 1e-5)
- `-d`, `--degeneracy-tolerance`: Frequency tolerance for degeneracy (default: 1e-4)
- `-l`, `--is-little-cogroup`: Use little co-group setting
- `-v`, `--log-level`: Verbosity level; 0=quiet, higher=more verbose (default: 0)
- `--show-verbose`: Also print full verbose irreps output (phonopy-style)
- `--verbose-file`: If set, write verbose output to this file instead of stdout
- `-b`, `--backend`: Backend driver to use for irrep identification. Choices: `"phonopy"` (default), `"irrep"`
- `-k`, `--kpname`: K-point name (e.g., `"GM"`, `"X"`, `"M"`) used by `irrep` backend

#### For `phonopy-irreps`:

- `-p`, `--params` (required): Path to phonopy params/YAML file
- `--qpoint` (required): Three floats for q-point in fractional coordinates
- `-s`, `--symprec`: Override symmetry precision. If omitted, anaddb_irreps will try to use the symmetry tolerance stored in the phonopy file, falling back to `1e-5`.
- `-d`, `--degeneracy-tolerance`: Frequency tolerance for degeneracy (default: 1e-4)
- `-l`, `--is-little-cogroup`: Use little co-group setting
- `-v`, `--log-level`: Verbosity level; 0=quiet, higher=more verbose (default: 0)
- `--show-verbose`: Also print full verbose irreps output (phonopy-style)
- `--verbose-file`: If set, write verbose output to this file instead of stdout
- `-b`, `--backend`: Backend driver to use for irrep identification. Choices: `"phonopy"` (default), `"irrep"`
- `-k`, `--kpname`: K-point name (e.g., `"GM"`, `"X"`, `"M"`) used by `irrep` backend

---

## Output Format

The CLI produces the same output format as the Python API, showing:

1. **Q-point coordinates** in fractional coordinates
2. **Space group** of the crystal
3. **Point group** of the structure at that q-point
4. **Mode table** with columns:
   - `qx, qy, qz`: q-point coordinates (repeated for each mode)
   - `band`: Mode index (0-based)
   - `freq(THz)`: Frequency in THz
   - `freq(cm-1)`: Frequency in cm⁻¹
   - `label`: Irreducible representation label (e.g., Eu, Eg, A1g, X1+, M2-)
   - `IR`: IR activity (`Y` = active, `.` = inactive)
   - `Raman`: Raman activity (`Y` = active, `.` = inactive)

**Note**: IR and Raman activity columns are shown for `phonopy` backend but omitted for `irrep` backend, as spectroscopic activity rules are typically defined only for the Γ point.

### Example Output

#### At Gamma point (phonopy backend):

```
q-point: [0.0000, 0.0000, 0.0000]
Space group: P-3m1
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

#### At non-Gamma point (irrep backend):

```
q-point: [0.0000, 0.5000, 0.0000]
Space group: Pm-3m
Point group: m-3m

# qx      qy      qz      band  freq(THz)   freq(cm-1)   label
  0.0000  0.5000  0.0000     0     -4.8804      -162.79  X5+       
  0.0000  0.5000  0.0000     1     -4.8804      -162.79  X5+       
  0.0000  0.5000  0.0000     2      3.3171       110.65  X5-       
  0.0000  0.5000  0.0000     3      3.3171       110.65  X5-       
```

Note the absence of IR/Raman columns in the non-Gamma output.

---

## Chiral Phase Transitions

The `phonopy-irreps` CLI automatically identifies phonon modes that can induce **chiral phase transitions**—transitions from a non-chiral parent space group to a chiral Sohncke space group. This is particularly useful for studying:

- **Multiferroic materials** where chirality couples to magnetic or electric order
- **Enantiomeric phase separation** in chiral crystals
- **Symmetry breaking pathways** to chiral ground states

### Additional Columns

When chiral transitions are possible for a given mode, two additional columns appear:

| Column | Description |
|--------|-------------|
| **OPD** | Order Parameter Direction in irrep space. Symbolic notation like `(a)`, `(a,0)`, `(a,a,...)` indicates the direction of the order parameter that breaks symmetry. |
| **Daughter SG** | Target chiral Sohncke space group(s) accessible via this transition, shown as `Symbol(#Number)`. Enantiomorphous pairs (Class II) are of particular interest. |

### Example: BaTiO₃ at Gamma

For cubic BaTiO₃ (Pm-3m, SG 221), the `T2u` mode can induce a transition to chiral R32:

```
# band  freq(THz)   freq(cm-1)   label(M)   label(BCS)  IR  Raman   OPD          Daughter SG
    9      8.5335       284.65  T2u         GM5-         .    .      (a,a,...)    R32(#155)
   10      8.5335       284.65  T2u         GM5-         .    .      (a,a,...)    R32(#155)
   11      8.5335       284.65  T2u         GM5-         .    .      (a,a,...)    R32(#155)
```

The `OPD: (a,a,...)` indicates the order parameter lies along the diagonal direction in the 3D irrep space, leading to the chiral subgroup R32.

### Example: TmFeO₃ at Gamma

For orthorhombic TmFeO₃ (Pnma, SG 62), multiple `Au` modes can induce transitions to the chiral group P2₁2₁2₁:

```
# band  freq(THz)   freq(cm-1)   label(M)   label(BCS)  IR  Raman   OPD          Daughter SG
    3      2.1839        72.85  Au          GM1-         .    .      (a)          P2_12_12_1(#19)
   12      4.8692       162.42  Au          GM1-         .    .      (a)          P2_12_12_1(#19)
   17      6.1103       203.82  Au          GM1-         .    .      (a)          P2_12_12_1(#19)
```

P2₁2₁2₁ (#19) is a **Class III Sohncke group**—a chiral space group that does not have a distinct enantiomorphous partner.

### Example: Class II Enantiomorphous Pairs

For space groups with transitions to **Class II** Sohncke groups, different OPDs lead to enantiomorphous partners. For example, from tetragonal P4₂/nmc (SG 131):

```
--- Class II (Enantiomorphous pairs) ---
  #   q-pt    Irrep    OPD      Daughter
  1    Z      Z4       (a,0)    P4_322(#95)
  2    Z      Z4       (0,a)    P4_122(#91)
```

Here, `(a,0)` leads to P4₃22 (#95) while `(0,a)` leads to its enantiomorph P4₁22 (#91). These two daughter phases are mirror images of each other—condensation of the same phonon with opposite OPD signs produces opposite handedness.

### When Chiral Transitions Appear

- **Columns hidden**: Parent space group is already chiral (Sohncke group), or no transitions found
- **Columns shown**: At least one mode can induce a transition to a chiral Sohncke group

### Interpreting OPD Notation

| OPD | Meaning |
|-----|---------|
| `(a)` | 1D irrep, single free parameter |
| `(a,0)` | 2D irrep, order parameter along first component |
| `(0,a)` | 2D irrep, order parameter along second component |
| `(a,a)` | 2D irrep, diagonal direction |
| `(a,a,...)` | Multi-D irrep, diagonal (isotropic) direction |

Different OPDs for the same irrep may lead to different daughter space groups.

### Further Reading

For the theoretical background on chiral transitions and Sohncke space groups, see `docs/chiral_transitions_theory.md`.
