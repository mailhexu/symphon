# Example 6: Circular Phonon Analysis

This example demonstrates circularly polarized phonon identification and analysis.

## Files

- `api_example.py` — Python API usage (self-contained, no data files needed)
- `cli_example.sh` — Command-line usage examples

## Quick Start

### Python API

```bash
cd 6_circular_phonon
python api_example.py
```

### Command Line

```bash
cd 6_circular_phonon
./cli_example.sh
```

## What This Example Shows

**1. SAM Calculator** — Spin Angular Momentum from displacement eigenvectors

The phonon SAM is defined as (Zhang & Niu, PRL 112, 085503 (2014)):
- **S** = ℏ Σ_κ Im(**ε**\*_κ × **ε**_κ)
- where **ε**_κ is the mass-weighted eigenvector (phonopy convention)
- Circularity = |**S**|/ℏ ∈ [0, 1]
  - 0 = linear, 1 = perfectly circular

**2. Abstract Finder** — Group-theory prediction (no phonon data needed)

Identifies multidimensional irreducible representations at special k-points.
These are the necessary condition for circular polarization — if a k-point
has only 1D irreps, no circular phonons are possible there.

**3. Concrete Finder** — Full analysis from ABINIT PHBST files

Requires a `_PHBST.nc` file. Constructs circular bases from degenerate
phonon modes using symmetry-adapted linear combinations.

## CLI Commands

```bash
# Group-theory candidates for a specific space group
symphon circular-abstract --sg 198

# Scan all 65 Sohncke (chiral) space groups
symphon circular-abstract --all-sohncke

# Full analysis from PHBST file
symphon-circular run_PHBST.nc
symphon-circular run_PHBST.nc --qpoint 0.5 0.0 0.0
symphon-circular run_PHBST.nc --symprec 1e-8 --verbose
```

## Key Concepts

| Term | Definition |
|------|-----------|
| **SAM** | Spin Angular Momentum — measures rotational character of phonon displacement |
| **Circularity** | \|S\|/ℏ — 0 (linear) to 1 (circular), intermediate = elliptical |
| **Sohncke group** | Space group without improper symmetries (65 total) — required for chiral phonons |
| **OPD** | Order Parameter Direction — the linear combination (e.g., (1, i)) giving circular polarization |
| **Multidim IR** | Irreducible representation with dim ≥ 2 — necessary for circular phonons |

## Requirements

```bash
pip install "symphon[irrep]"
```

For PHBST file support (concrete finder):

```bash
pip install "symphon[abipy,irrep]"
```
