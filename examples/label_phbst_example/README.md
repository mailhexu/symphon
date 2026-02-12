# Label PHBST Example

This directory demonstrates how to use `anaddb_irreps` to label phonon modes from ABINIT's PHBST NetCDF output.

## Files

- `LAO_PHBST.nc`: Example PHBST file containing phonon data for LAO (LaAlO3) structure along Gamma-X-M path (172 q-points)
- `label_example.py`: Python script demonstrating how to analyze file
- `mode_labels.txt`: Output file (generated when running script)
- `.gitignore`: Git ignore file to exclude generated output

## Running the Example

```bash
python label_example.py
```

This will analyze three high-symmetry points:

1. **Gamma point** (index 0, q = [0, 0, 0])
   - Uses `phonopy` backend
   - Shows Mulliken notation labels (T1u, T2u, A1g, etc.)
   - Includes IR and Raman activity columns
   
2. **X point** (index 20, q = [0, 0.5, 0])
   - Uses `irrep` backend
   - Shows BCS-style labels (X5-, X5+, X2+, etc.)
   - No IR/Raman activity (defined only for Γ point)
   
3. **M point** (index 40, q = [0.5, 0.5, 0])
   - Uses `irrep` backend
   - Shows BCS-style labels (M2+, M5-, M1+, etc.)
   - No IR/Raman activity (defined only for Γ point)

Results are saved to `mode_labels.txt`.

## Understanding the Output

The output for each point includes:
- **q-point coordinates** in fractional coordinates
- **Space group** (Pm-3m for LAO)
- **Point group** (m-3m for LAO)
- **Mode table** with columns:
  - `qx, qy, qz`: q-point coordinates
  - `band`: Mode index (0-based)
  - `freq(THz)`: Frequency in THz
  - `freq(cm-1)`: Frequency in cm⁻¹
  - `label`: Irreducible representation label
  - `IR`: IR activity (`Y` = active, `.` = inactive) - only for Gamma point
  - `Raman`: Raman activity (`Y` = active, `.` = inactive) - only for Gamma point

## Customization

To analyze different q-points:

```python
points_to_analyze = [
    {'name': 'Gamma', 'index': 0,   'qcoords': [0.0, 0.0, 0.0], 'backend': 'phonopy'},
    {'name': 'X',     'index': 20,  'qcoords': [0.0, 0.5, 0.0], 'backend': 'irrep',  'kpname': 'X'},
    {'name': 'M',     'index': 40,  'qcoords': [0.5, 0.5, 0.0], 'backend': 'irrep',  'kpname': 'M'},
]
```

To use with your own PHBST file:

```python
phbst_fname = 'path/to/your_PHBST.nc'
```

## Notes

- The LAO_PHBST.nc file contains 172 q-points along a path from Gamma to X to M
- Gamma point analysis uses phonopy backend (optimal for Γ-point)
- Non-Gamma point analysis uses irrep backend (required for k-points other than Γ)
- The irrep backend requires installation: `pip install "anaddb_irreps[irrep]"`
