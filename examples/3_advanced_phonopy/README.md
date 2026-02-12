# Example 3: Advanced Phonopy Analysis

This example demonstrates advanced features using TmFeO3 (orthorhombic Pnma structure).

## Files

- `TmFeO3_phonopy_params.yaml` - Phonopy params file for TmFeO3
- `raman_ir_analysis.py` - *(To be created)* Analysis script focusing on spectroscopic activity

## Coming Soon

This directory is prepared for advanced examples showing:
- IR and Raman activity analysis in detail
- Comparison of different space groups
- Custom analysis workflows

For now, you can use the basic phonopy-irreps CLI:

```bash
phonopy-irreps --params TmFeO3_phonopy_params.yaml
```

## Notes

- TmFeO3 has orthorhombic Pnma symmetry (different from cubic BaTiO3)
- Different space groups have different high-symmetry points
- The automatic discovery feature works for any space group
