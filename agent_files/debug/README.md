# Debug Directory

This directory contains test and debug scripts for development.

## Files

- `test_tmfeo3.sh` - Test script for TmFeO3 phonopy irrep analysis
- `test_tmfeo3.py` - Python wrapper (deprecated - use bash script instead)
- `debug_kpoint_transform.py` - Debug script to understand k-point issues
- `debug_kpoint_transform2.py` - Find transformation matrix candidates
- `find_transformation.py` - Find exact transformation for Pnma
- `README.md` - This file

## How to Run

```bash
# Test with BaTiO3 (works perfectly - cubic system)
uv run phonopy-irreps --params examples/2_basic_phonopy/BaTiO3_phonopy_params.yaml

# Test with TmFeO3 (all points work now!)
bash agent_files/debug/test_tmfeo3.sh
```

## Recent Fixes

✅ **Fixed atom mapping tolerance** - Changed hardcoded `1e-5` to use `self._symprec` in irrep_backend.py. This fixes the "Atom mapping failed" error at Gamma point.

✅ **Fixed k-point transformation for Pnma (SG 62)** - irreptables returns k-points in reference UC coordinates, but the irrep backend expects primitive UC coordinates. Added transformation: `k_prim = (k_z, k_x, k_y)` for space group 62.

## Test Status

✅ **BaTiO3** - All high-symmetry points work (Pm-3m, cubic)  
✅ **TmFeO3** - All 8 high-symmetry points work (Pnma, orthorhombic)
