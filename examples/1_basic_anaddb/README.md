# Example 1: Basic anaddb-irreps Usage

This example demonstrates basic usage of `anaddb-irreps` with ABINIT's PHBST NetCDF output files.

## Files

- `MoS2_1T/run_PHBST.nc` - PHBST file for MoS2 1T structure
- `MoS2_1T/anaddb_input/` - Original anaddb input files (for reference)
- `api_example.py` - Python API usage example
- `cli_example.sh` - Command-line usage example

## Quick Start

### Command Line

```bash
cd 1_basic_anaddb
./cli_example.sh
```

Or directly:
```bash
anaddb-irreps --phbst MoS2_1T/run_PHBST.nc --q-index 0
```

### Python API

```bash
cd 1_basic_anaddb
python api_example.py
```

## What This Example Shows

- How to analyze phonon modes from PHBST files
- Irreducible representation labeling at Gamma point
- IR and Raman activity identification
- Space group and point group information

## Output

The tool prints a concise table with:
- Mode frequencies (THz and cm⁻¹)
- Irrep labels (Mulliken notation: T1u, T2u, etc.)
- IR/Raman activity markers
- Space group and point group symmetry

## Notes

- PHBST files require explicit q-point index (use `--q-index`)
- For non-Gamma points, add `--backend irrep --kpname X` (where X is the k-point name)
- The MoS2 1T example analyzes the Gamma point (index 0)
