# anaddb_irreps
A simple wrapper of the phonopy irreps module for finding irreduciple representations of the phonon modes in anaddb output.

## Installation

Install from PyPI:

```
pip install anaddb_irreps
```

Or from source:

```
pip install .
```

## Usage (Python API):

- Step 1:

  Run anaddb to get the phbst file with phonon frequencies and eigen vectors inside. See for example in the examples/MoS2_1T/anaddb_input directory. 

- Step 2:

  Make a python script with the content similar to below and run it. 

  

```python
from anaddb_irreps import IrRepsAnaddb

def show_phbst_irreps():
    irr = IrRepsAnaddb(
        phbst_fname="run_PHBST.nc",
        ind_q=0,
        symprec=1e-5,
        degeneracy_tolerance=1e-4)
    irr.run()
    irr._show(True)

show_phbst_irreps()
```

The parameters in the functions:

 *      phbst_fname: name of phbst file.
 *      ind_q: index of the q point in the phbst.
 *      is_little_cogroup: 
 *      symprec: precision for deciding the symmetry of the atomic structure.
 *      degeneracy_tolenrance: the tolerance of frequency difference in deciding the degeneracy.
 *      log_level: how much information is in the output. 
 
## Usage (CLI):

You can use the `anaddb-irreps` command-line tool as a lightweight
alternative to writing a Python script.

Basic example (equivalent to the Python example above):

```bash
anaddb-irreps \
  --phbst run_PHBST.nc \
  --q-index 0 \
  --symprec 1e-5 \
  --degeneracy-tolerance 1e-4
```

Options:

- `-p`, `--phbst` (required): Path to PHBST NetCDF file (e.g. `run_PHBST.nc`).
- `-q`, `--q-index` (required): Index of the q-point in the PHBST file (0-based).
- `-s`, `--symprec`: Symmetry precision passed to phonopy (default: `1e-5`).
- `-d`, `--degeneracy-tolerance`: Frequency tolerance for degeneracy detection (default: `1e-4`).
- `-l`, `--is-little-cogroup`: Use little co-group setting.
- `-v`, `--log-level`: Verbosity level passed through to `IrRepsAnaddb` (default: `0`).

The CLI prints the same irreps information as the Python API example.


