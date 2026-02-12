# Label PHBST Example

This directory demonstrates how to use `anaddb_irreps` to label phonon modes from ABINIT's PHBST NetCDF output.

## Files

- `LAO_PHBST.nc`: Example PHBST file containing phonon data for LAO (LaAlO3) structure along Gamma-X path (172 q-points)
- `label_example.py`: Python script demonstrating how to analyze the file
- `mode_labels.txt`: Output file (generated when running the script)
- `.gitignore`: Git ignore file to exclude generated output

## Running the Example

```bash
python label_example.py
```

This will:
1. Read the PHBST file and identify available q-points
2. Analyze the Gamma point (q-point index 0) using the phonopy backend
3. Print the irreducible representation table with frequencies and IR/Raman activity
4. Save results to `mode_labels.txt`

## Understanding the Output

The output includes:
- **q-point coordinates** in fractional coordinates
- **Space group** and **point group** of the crystal
- **Mode table** with columns:
  - `qx, qy, qz`: q-point coordinates
  - `band`: Mode index (0-based)
  - `freq(THz)`: Frequency in THz
  - `freq(cm-1)`: Frequency in cm⁻¹
  - `label`: Irreducible representation label (e.g., A1g, Eu, T1u)
  - `IR`: IR activity (`Y` = active, `.` = inactive)
  - `Raman`: Raman activity (`Y` = active, `.` = inactive)

## Modifying the Example

To analyze a different q-point, change the `ind_q` variable in `label_example.py`:

```python
# Analyze a different q-point
ind_q = 20  # Change to desired index (0 to 171)
```

To use with your own PHBST file:
```python
phbst_fname = 'path/to/your_PHBST.nc'
```

## Notes

- This example uses the `phonopy` backend, which is best suited for Gamma-point analysis
- For non-Gamma points, you would need to use the `irrep` backend and specify the k-point name
- The LAO_PHBST.nc file contains a dense q-point grid; this example uses only the Gamma point for clarity
