# symphon Examples

This directory contains organized examples demonstrating different features of `symphon`.

## Directory Structure

### 1. Basic anaddb-irreps (`1_basic_anaddb/`)
**Basic usage with ABINIT PHBST files**
- Input: PHBST NetCDF files from anaddb
- Example: MoS2 1T structure
- Features: Gamma point analysis, IR/Raman activity
- **Start here** if you're using ABINIT/anaddb output

### 2. Basic phonopy-irreps (`2_basic_phonopy/`)
**NEW: Automatic high-symmetry point discovery**
- Input: Phonopy params/YAML files
- Example: BaTiO3 (cubic Pm-3m)
- Features: 
  - Auto-discovers all high-symmetry k-points
  - Dual labels at Gamma (Mulliken + BCS)
  - BCS labels at other points
- **Start here** for the simplest workflow

### 3. Advanced phonopy (`3_advanced_phonopy/`)
**Advanced analysis examples**
- Input: TmFeO3 phonopy params (orthorhombic Pnma)
- Features: Different space group, spectroscopic analysis
- Status: Prepared for future examples

### 4. Multi-kpoint anaddb (`4_multi_kpoint_anaddb/`)
**Multiple k-points along a path**
- Input: LAO PHBST file with 172 q-points
- Example: Analysis of Gamma, X, and M points
- Features: Both backends (phonopy + irrep), path analysis

## Quick Start

### For ABINIT users:
```bash
cd 1_basic_anaddb
./cli_example.sh
```

### For phonopy users (RECOMMENDED - simplest!):
```bash
cd 2_basic_phonopy
./cli_example.sh
```

This single command automatically analyzes **all** high-symmetry points!

## Feature Comparison

| Feature | anaddb-irreps | phonopy-irreps |
|---------|---------------|----------------|
| **Auto k-point discovery** | ❌ No (need explicit index) | ✅ **Yes** (automatic!) |
| **Dual labels at Gamma** | Optional flag | ✅ **Automatic** |
| **Input file** | PHBST NetCDF | Phonopy YAML/params |
| **k-point specification** | Index-based | Automatic discovery |
| **Best for** | ABINIT workflows | General phonopy use |
| **CLI command** | `symphon anaddb-irreps` | `symphon phonopy-irreps` |

## Requirements

**Basic features:**
```bash
pip install symphon
```

**Auto-discovery & non-Gamma points:**
```bash
pip install "symphon[irrep]"
```

**ABINIT PHBST support:**
```bash
pip install "symphon[abipy]"
```

**Everything:**
```bash
pip install "symphon[irrep,abipy]"
```

## What's New

The `phonopy-irreps` CLI automatically discovers and analyzes all high-symmetry
points with optimal settings — no need to specify q-points, backends, or label
options manually:

```bash
# Just run it!
phonopy-irreps --params file.yaml
```

✨ Automatically discovers and analyzes **all** high-symmetry points!

## See Also

- Main README: `../README.md`
- Usage guide: `../docs/usage.md`
- Irreps theory: `../docs/irreps_guide.md`
- IR/Raman activity: `../docs/activity_methodology.md`
