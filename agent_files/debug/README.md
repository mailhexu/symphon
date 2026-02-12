# Debug Directory

This directory contains test and debug scripts for development.

## Files

- `test_tmfeo3.sh` - Test script for TmFeO3 phonopy irrep analysis
- `test_tmfeo3.py` - Python wrapper (deprecated - use bash script instead)
- `README.md` - This file

## How to Run

```bash
# Test with BaTiO3 (works perfectly - cubic system)
uv run phonopy-irreps --params examples/2_basic_phonopy/BaTiO3_phonopy_params.yaml

# Test with TmFeO3 (GM and R points work, S/T/U/X/Y/Z have k-point transformation issues)
bash agent_files/debug/test_tmfeo3.sh
```

## Recent Fixes

✅ **Fixed atom mapping tolerance** - Changed hardcoded `1e-5` to use `self._symprec` in irrep_backend.py. This fixes the "Atom mapping failed" error at Gamma point for TmFeO3.

## Known Issues

**TmFeO3 k-point transformation**: For orthorhombic systems like Pnma, the irrep package uses different coordinate systems (reference UC vs primitive UC) for some k-points. Points like S, T, U have coordinate mismatches between what `irreptables` returns and what the irrep backend expects. This is a limitation of the irrep package, not our code.

## Test Status

✅ **BaTiO3** - All high-symmetry points work (Pm-3m, cubic)  
⚠️ **TmFeO3** - GM and R work, S/T/U/X/Y/Z have k-point transformation issues (Pnma, orthorhombic)
