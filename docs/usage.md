# Usage Guide

This guide provides detailed usage instructions for `symphon` including both Python API and command-line interfaces.

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

Run anaddb to get the PHBST file with phonon frequencies and eigenvectors. See example in `examples/1_basic_anaddb/MoS2_1T/anaddb_input/`.

#### Basic Usage

```python
from symphon import print_irreps

# Simple usage
print_irreps("run_PHBST.nc", ind_q=0)
```

#### With Options

```python
from symphon import print_irreps

print_irreps(
    "run_PHBST.nc",
    ind_q=0,
    symprec=1e-8,              # Symmetry precision (default: 1e-5)
    degeneracy_tolerance=1e-4, # Frequency tolerance (default: 1e-4)
    log_level=0,               # Verbosity: 0=quiet, 1+=verbose (default: 0)
    show_verbose=True,         # Show detailed phonopy output (default: False)
)
```

**Note**: The backend is automatically selected based on the q-point (phonopy for Gamma, irrep for others).

### From phonopy params/YAML

If you already have a phonopy params/YAML file (e.g. `phonopy_params.yaml` or `phonopy.yaml`), you can use the phonopy-based helper:

#### Basic Usage

```python
from symphon import print_irreps_phonopy

# Simple usage at any q-point (backend auto-selected)
print_irreps_phonopy("phonopy_params.yaml", qpoint=[0.0, 0.0, 0.0])
```

#### With Options

```python
from symphon import print_irreps_phonopy

print_irreps_phonopy(
    "phonopy_params.yaml",
    qpoint=[0.5, 0.5, 0.0],  # M point
    symprec=1e-5,
    degeneracy_tolerance=1e-4,
    log_level=0,
    show_verbose=False,
)
```

**Note**: The backend is automatically selected, and k-point names are determined from the q-point coordinates.

### Parameters

#### For `print_irreps` (anaddb route):

- **phbst_fname** (str, required): Path to PHBST NetCDF file
- **ind_q** (int, required): Index of q-point in PHBST file (0-based)
- **symprec** (float): Symmetry precision for structure analysis (default: 1e-5)
- **degeneracy_tolerance** (float): Frequency tolerance for degeneracy detection (default: 1e-4)
- **log_level** (int): Verbosity level; 0=quiet, higher=more verbose (default: 0)
- **show_verbose** (bool): Show detailed phonopy irreps output (default: False)

#### For `print_irreps_phonopy` (phonopy route):

- **phonopy_params** (str, required): Path to phonopy params/YAML file
- **qpoint** (sequence of 3 floats, required): q-point in fractional coordinates
- **symprec** (float or None): Symmetry precision for structure analysis. If `None` (or omitted), symphon will try to use the symmetry tolerance recorded in the phonopy file (e.g., `phonopy.symmetry_tolerance` in the YAML), falling back to `1e-5` when not available.
- **degeneracy_tolerance** (float): Frequency tolerance for degeneracy detection (default: 1e-4)
- **log_level** (int): Verbosity level; 0=quiet, higher=more verbose (default: 0)
- **show_verbose** (bool): Show detailed phonopy irreps output (default: False)

---

## Command-Line Interface

The `symphon` package provides a unified CLI with subcommands, plus a standalone command for backward compatibility.

### Unified CLI (Recommended)

The `symphon` command provides all functionality through subcommands:

```bash
symphon <subcommand> [options]
```

Available subcommands:
- `anaddb-irreps` - Irreps from anaddb PHBST file
- `phonopy-irreps` - Irreps from phonopy params/YAML file (auto-discovery)
- `find-chiral-transition` - Find chiral phase transitions
- `magnetic-chiral` - Find chiral magnetic transitions
- `abstract-magnetic` - Abstract chiral magnetic transitions by space group
- `msg` - Identify chirality of a Magnetic Space Group

### Legacy Standalone Commands

- `anaddb-irreps` - Same as `symphon anaddb-irreps` (for backward compatibility)
- `phonopy-irreps` - Same as `symphon phonopy-irreps` (standalone entry point)

---

## Phonon Irreps Commands

### anaddb-irreps

Analyze phonon irreps from anaddb PHBST NetCDF files.

#### Auto-Discovery Mode (Default)

```bash
# Automatically find and analyze all high-symmetry q-points
symphon anaddb-irreps --phbst run_PHBST.nc

# Or using standalone command:
anaddb-irreps --phbst run_PHBST.nc
```

#### Single Q-Point Mode

```bash
# Analyze specific q-point (0-based index)
symphon anaddb-irreps --phbst run_PHBST.nc --q-index 0

# For non-Gamma points, specify k-point name
symphon anaddb-irreps --phbst run_PHBST.nc --q-index 5 --kpname M
```

#### With Options

```bash
symphon anaddb-irreps \
  --phbst run_PHBST.nc \
  --q-index 0 \
  --symprec 1e-8 \
  --degeneracy-tolerance 1e-4 \
  --log-level 1 \
  --show-verbose
```

### phonopy-irreps

Analyze phonon irreps from phonopy params/YAML files. **Always runs in auto-discovery mode**, automatically analyzing all high-symmetry k-points using the `irrep` backend.

#### Basic Usage

```bash
# Automatically analyze all high-symmetry k-points
symphon phonopy-irreps --params phonopy_params.yaml

# Or using standalone command:
phonopy-irreps --params phonopy_params.yaml
```

#### With Options

```bash
symphon phonopy-irreps \
  --params phonopy_params.yaml \
  --symprec 1e-5 \
  --degeneracy-tolerance 1e-4 \
  --log-level 0 \
  --chiral
```

#### Chiral Transitions

Add `--chiral` flag to show possible chiral symmetry-breaking transitions:

```bash
symphon phonopy-irreps --params phonopy_params.yaml --chiral
```

This adds two columns to the output:
- **OPD**: Order Parameter Direction (e.g., `(a)`, `(a,0)`, `(a,a,...)`)
- **Daughter SG**: Target chiral Sohncke space group(s)

### CLI Options

#### For `anaddb-irreps`:

- `-p`, `--phbst` (required): Path to PHBST NetCDF file
- `-q`, `--q-index` (optional): Index of q-point in PHBST file (0-based). If omitted, analyzes all high-symmetry q-points automatically.
- `-s`, `--symprec`: Symmetry precision (default: 1e-5)
- `-d`, `--degeneracy-tolerance`: Frequency tolerance for degeneracy (default: 1e-4)
- `-v`, `--log-level`: Verbosity level; 0=quiet, higher=more verbose (default: 0)
- `--show-verbose`: Also print full verbose irreps output (phonopy-style)
- `--verbose-file`: If set, write verbose output to this file instead of stdout
- `-k`, `--kpname`: K-point name (e.g., `"GM"`, `"X"`, `"M"`). Used only in single q-point mode; in auto-discovery mode, labels are determined automatically.

#### For `phonopy-irreps`:

- `-p`, `--params` (required): Path to phonopy params/YAML file
- `-s`, `--symprec`: Override symmetry precision. If omitted, uses the symmetry tolerance from the phonopy file, falling back to `1e-5`.
- `-d`, `--degeneracy-tolerance`: Frequency tolerance for degeneracy (default: 1e-4)
- `-v`, `--log-level`: Verbosity level; 0=quiet, higher=more verbose (default: 0)
- `--show-verbose`: Also print full verbose irreps output (phonopy-style)
- `--verbose-file`: If set, write verbose output to this file instead of stdout
- `--chiral`: Show chiral transition information (OPD and daughter space groups)

**Note**: The `phonopy-irreps` command automatically analyzes all high-symmetry k-points using the `irrep` backend. There is no `--qpoint` or `--backend` option - the analysis is fully automatic.

---

## Chiral Transitions Commands

### find-chiral-transition

Find possible chiral phase transitions from a parent space group to chiral Sohncke subgroups.

#### Basic Usage

```bash
# Find transitions from a specific space group
symphon find-chiral-transition --sg 136

# Find transitions from all non-Sohncke groups (with caching)
symphon find-chiral-transition --all

# Filter by daughter space group
symphon find-chiral-transition --sg 84 --daughter 77
```

#### Options

- `--sg`, `--space-group`: Space group number (1-230) to find transitions FROM
- `--daughter`: Filter transitions by daughter space group number
- `--all`: Print summary for all non-Sohncke space groups
- `--list-daughters`: List all possible chiral daughter space groups
- `--list-sons`: List all space groups that can transition to a chiral daughter
- `--output`, `-o`: Output file (default: stdout)
- `--verbose`, `-v`: Verbose output
- `--cache`: Use cached results for `--all` (default: True)
- `--no-cache`: Disable caching for `--all`
- `--refresh-cache`: Force refresh the cache for `--all`

---

## Magnetic Chirality Commands

### magnetic-chiral

Find chiral magnetic phase transitions from a structure with magnetic sites.

```bash
symphon magnetic-chiral \
  --structure POSCAR \
  --qpoint 0 0 0.5 \
  --mag-sites 0,1
```

#### Options

- `--structure` (required): Path to structural file (e.g., POSCAR, CIF)
- `--qpoint`: Propagation wavevector in fractional coordinates (default: `0 0 0`)
- `--mag-sites` (required): Comma-separated list of magnetic atom indices (0-based)
- `--symprec`: Symmetry precision (default: 1e-5)

### abstract-magnetic

Find abstract chiral magnetic transitions by space group number.

```bash
symphon abstract-magnetic --spg 221 --qpoint 0 0 0
```

#### Options

- `--spg` (required): Parent space group number (1-230)
- `--qpoint`: Propagation wavevector in fractional coordinates (default: `0 0 0`)
- `--multi-k`: Include multi-k transitions
- `--all`: Show all transitions, not just chiral ones
- `--symprec`: Symmetry precision (default: 1e-5)

### msg

Identify the chirality classification of a Magnetic Space Group (MSG).

```bash
symphon msg 18.16
```

#### Arguments

- `identifier` (required): BNS number (e.g., `'18.16'`) or UNI number (e.g., `114`) of the Magnetic Space Group

This command displays:
- BNS and UNI numbers
- Family space group
- Magnetic type (I, II, III, or IV)
- Chirality status
- Sohncke class (I, II, or III)
- Enantiomorph partner (for Class II groups)

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

**Note**: IR and Raman activity columns are shown at the Gamma point but omitted for non-Gamma points, as spectroscopic activity rules are only defined at Γ.

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

The **Order Parameter Direction (OPD)** describes the direction of symmetry breaking in the space of irreducible representations. Understanding OPD notation is essential for predicting which daughter crystal structures are accessible from a given phonon mode.

#### What is an Order Parameter?

In Landau theory of phase transitions, an **order parameter** is a physical quantity that is zero in the high-symmetry phase and non-zero in the low-symmetry phase. For structural phase transitions driven by phonons:

- The order parameter components correspond to the **amplitude of atomic displacement patterns** associated with each basis vector of the irreducible representation
- The **dimension** of the order parameter equals the dimension of the irrep
- Different **directions** in this parameter space lead to different symmetry-breaking patterns

#### OPD Notation Explained

| OPD | Meaning | Example Physical Interpretation |
|-----|---------|--------------------------------|
| `(a)` | 1D irrep, single free parameter | The mode condenses with arbitrary amplitude `a`. Only one daughter group possible. |
| `(a,0)` | 2D irrep, first component only | Only the first displacement pattern condenses; second pattern has zero amplitude. |
| `(0,a)` | 2D irrep, second component only | Only the second displacement pattern condenses. Often leads to enantiomorphous partner of `(a,0)`. |
| `(a,a)` | 2D irrep, diagonal direction | Both displacement patterns condense with equal amplitude. |
| `(a,-a)` | 2D irrep, anti-diagonal | Both patterns condense but with opposite phases. May give different daughter than `(a,a)`. |
| `(a,a,...)` | Multi-D irrep, all components equal | Diagonal direction in high-dimensional space. Often maximizes symmetry reduction. |
| `(a,b,c,d)` | Multi-D irrep, distinct values | Complex displacement pattern with multiple independent amplitudes. |

Here, `a`, `b`, `c`, `d` represent **free parameters** (real numbers) that can take any non-zero value. The actual magnitude determines how far the structure is from the transition point, but the **direction** (ratios between components) determines which daughter symmetry results.

#### Key Concepts

1. **Equivalent OPDs**: OPDs that differ only by an overall scale factor are equivalent:
   - `(a,0)` ≡ `(2a,0)` ≡ `(-a,0)` (same direction)
   - These all lead to the same daughter group

2. **Collinear OPDs**: For 1D irreps, all OPDs are collinear (same direction), so:
   - Only one OPD direction exists: `(a)`
   - Only one daughter group possible (per irrep)

3. **Non-collinear OPDs**: For multi-D irreps, different directions give different daughters:
   - `(a,0)` and `(0,a)` are perpendicular directions → often different daughters
   - `(a,a)` is diagonal → may give yet another daughter

4. **Enantiomorphous pairs**: For Class II Sohncke groups:
   - `(a,0)` might give P4₃22 (right-handed screw)
   - `(0,a)` gives P4₁22 (left-handed screw)
   - The handedness is determined by which component condenses

#### Example: 2D Irrep with Multiple Daughters

Consider a hypothetical 2D irrep E at a zone boundary:

```
OPD (a,0)  →  Daughter: P2₁ (screw along x)
OPD (0,a)  →  Daughter: P2₁ (screw along y)  
OPD (a,a)  →  Daughter: C2 (glide+rotation)
```

All three are valid symmetry-breaking patterns from the same phonon mode, but the **direction** of the order parameter determines which atomic displacement pattern actually occurs.

#### Physical Interpretation

The OPD components describe how atoms move during the phase transition:

- **First component**: Amplitude of displacement pattern 1
- **Second component**: Amplitude of displacement pattern 2
- **...**

For a 2D irrep with basis functions ψ₁ and ψ₂, the total displacement is:

```
δr = a·ψ₁ + b·ψ₂
```

where `(a,b)` is the OPD. Different choices of `a` and `b` give different displacement patterns, which may have different residual symmetries.

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
