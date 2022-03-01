# anaddb_irreps
A simple wrapper of the phonopy irreps module for finding irreduciple representations of the phonon modes in anaddb output

## Installation
```
    pip install . 
```

## Usage:

- Step 1:

  Run anaddb to the phbst file with phonon frequencies and eigen vectors inside. See for example in the examples/MoS2_1T/anaddb_input directory. 

- Step 2:

  Make a python script with the cotent similar to below.

  

```python
from anaddb_irreps import IrRepsAnaddb

def show_phbst_irreps():
    irr = IrRepsAnaddb(
        phbst_fname="run_PHBST.nc",
        ind_q=0,
        symprec=1e-8,
        degeneracy_tolerance=1e-4)
    irr.run()
    irr._show(True)

show_phbst_irreps()
```

The parameters in the functions:

 *      phbst_fname: name of phbst file.
 *      ind_q: index of the q point in the phbst.
 *     is_little_cogroup: 
 *     symprec: precision for deciding the symmetry of the atomic structure.
 *     degeneracy_tolenrance: the tolerance of frequency difference in deciding the degeneracy.
 *    log_level: how much information is in the output. 
