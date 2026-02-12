# Example 2: Basic phonopy-irreps Usage

This example demonstrates the new automatic high-symmetry point analysis with `phonopy-irreps`.

## Files

- `BaTiO3_phonopy_params.yaml` - Phonopy params file for BaTiO3 (cubic Pm-3m)
- `api_example.py` - Python API usage example (single q-point)
- `cli_example.sh` - **NEW** simplified CLI (auto-discovers all high-symmetry points)
- `cli_example_old.sh` - Old explicit CLI usage (for reference)

## Quick Start

### Command Line (NEW - Recommended!)

```bash
cd 2_basic_phonopy
./cli_example.sh
```

Or directly:
```bash
phonopy-irreps --params BaTiO3_phonopy_params.yaml
```

This **automatically**:
- Discovers all high-symmetry k-points (GM, M, R, X)
- Shows **dual labels** at Gamma (Mulliken + BCS) with IR/Raman activity
- Shows BCS labels at other points

### Python API (Manual q-point specification)

```bash
cd 2_basic_phonopy
python api_example.py
```

## What This Example Shows

- **Automatic high-symmetry point discovery** (CLI only)
- **Dual-label display at Gamma**: Mulliken (T1u, T2u) + BCS (GM4-, GM5-)
- BCS labels at non-Gamma points (M, R, X)
- IR and Raman activity at Gamma point
- Space group symmetry analysis

## Example Output

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
...

# M point (k=(0.5, 0.5, 0.0))
============================================================
q-point: [0.5000, 0.5000, 0.0000]
...
# qx      qy      qz      band  freq(THz)   freq(cm-1)   label
 0.5000  0.5000  0.0000     0     -3.9928      -133.19  M2-
...
```

## Notes

- **CLI auto-discovery** requires `pip install "anaddb_irreps[irrep]"`
- Python API still requires explicit qpoint specification
- Dual labels (Mulliken + BCS) only shown at Gamma point
- IR/Raman activity only available at Gamma point
