# Irreducible Representation Labeling Guide

This guide describes the mathematical foundations and usage of the irrep labeling capabilities in `anaddb_irreps`, particularly for non-Gamma ($q \neq 0$) phonons.

## 1. Notation and Labels

### Mulliken Notation
Phonon modes are labeled using Mulliken notation based on the symmetry of their eigenvectors:
- **A**: Non-degenerate (1D), symmetric with respect to the principal rotation axis.
- **B**: Non-degenerate (1D), anti-symmetric with respect to the principal rotation axis.
- **E**: Doubly degenerate (2D).
- **T** (or **F**): Triply degenerate (3D).

**Subscripts and Superscripts:**
- **g** (*gerade*): Symmetric with respect to inversion.
- **u** (*ungerade*): Anti-symmetric with respect to inversion.
- **1/2**: Symmetric/Anti-symmetric with respect to a perpendicular $C_2$ axis or vertical mirror plane.
- **' / ''**: Symmetric/Anti-symmetric with respect to a horizontal mirror plane.

### K-point Labels
High-symmetry points in the Brillouin zone are labeled following the conventions of the **Bilbao Crystallographic Server (BCS)**:
Take a simple cubic cell for example, the high-symmetry points are:
- **GM**: $\Gamma$ point $(0, 0, 0)$
- **X**: $(0, 1/2, 0)$ or $(1/2, 0, 0)$ depending on the lattice.
- **M**: $(1/2, 1/2, 0)$
- **R**: $(1/2, 1/2, 1/2)$

---

## 2. Mathematical Framework

### Backend 1: Phonopy (Standard $\Gamma$-point)
At the $\Gamma$ point, the symmetry is governed by the full point group of the crystal. The character of the representation for a mode is simply:

$$\chi(R) = \text{Tr}(\mathbf{R})$$

where $\mathbf{R}$ is the 3x3 rotation matrix. The label is found by matching the calculated character table of the phonon modes against the hardcoded character tables in `phonopy`.

### Backend 2: Irrep Package (Arbitrary $q$)
For non-Gamma points, the symmetry is governed by the **Little Group of $q$**, denoted $G_q$, which consists of all space group operations $\{R|\boldsymbol{\tau}\}$ that leave $q$ invariant modulo a reciprocal lattice vector $\mathbf{K}$:

$$\mathbf{R}\mathbf{q} = \mathbf{q} + \mathbf{K}$$

#### Transformation of Displacement Vectors
The phonon displacement vector $\mathbf{u}_k(\mathbf{q})$ for atom $k$ transforms under an operation $\{R|\boldsymbol{\tau}\}$ as:
$$T_{\{R|\boldsymbol{\tau}\}} \mathbf{u}_k(\mathbf{q}) = e^{-i \mathbf{q} \cdot \mathbf{G}_{jk}} \mathbf{R} \mathbf{u}_j(\mathbf{q})$$
where:
- $\mathbf{R}$ is the Cartesian rotation matrix.
- Atom $j$ is mapped to atom $k$ by the symmetry operation: $\mathbf{r}_k = \mathbf{R}\mathbf{r}_j + \boldsymbol{\tau} - \mathbf{G}_{jk}$.
- $\mathbf{G}_{jk}$ is a lattice vector required to bring the transformed position back into the primary unit cell.

#### Multiplicity and Identification
The total trace $\chi_{block}(R|\boldsymbol{\tau})$ of a degenerate block of modes is calculated. The multiplicity $n$ of a specific irrep $\Gamma_i$ within that block is determined by the projection formula:
$$n_i = \frac{1}{g} \sum_{\{R|\boldsymbol{\tau}\} \in G_q} \chi_i(R|\boldsymbol{\tau})^* \chi_{block}(R|\boldsymbol{\tau})$$
where $g$ is the order of the little group. If $n_i \approx 1$, the block is successfully identified as belonging to irrep $\Gamma_i$.

---

## 3. Tutorial: BaTiO3 Example

BaTiO3 in its cubic phase (Space Group 221, $Pm\bar{3}m$) is an ideal test case.

### Gamma Point ($\Gamma$)
The `phonopy` backend is the default and is best suited for the $\Gamma$ point:
```bash
phonopy-irreps --params BaTiO3_phonopy_params.yaml --qpoint 0 0 0
```

**Raw CLI Output:**
```text
q-point: [0.0000, 0.0000, 0.0000]
Point group: m-3m

# qx      qy      qz      band  freq(THz)   freq(cm-1)   label        IR  Raman
 0.0000  0.0000  0.0000     0     -6.0487      -201.76  T1u          Y    .  
 0.0000  0.0000  0.0000     1     -6.0487      -201.76  T1u          Y    .  
 0.0000  0.0000  0.0000     2     -6.0487      -201.76  T1u          Y    .  
 0.0000  0.0000  0.0000     3     -0.0000        -0.00  T1u          Y    .  
 0.0000  0.0000  0.0000     4      0.0000         0.00  T1u          Y    .  
 0.0000  0.0000  0.0000     5      0.0000         0.00  T1u          Y    .  
 0.0000  0.0000  0.0000     6      5.3224       177.54  T1u          Y    .  
 0.0000  0.0000  0.0000     7      5.3224       177.54  T1u          Y    .  
 0.0000  0.0000  0.0000     8      5.3224       177.54  T1u          Y    .  
 0.0000  0.0000  0.0000     9      8.5335       284.65  T2u          .    .  
 0.0000  0.0000  0.0000    10      8.5335       284.65  T2u          .    .  
 0.0000  0.0000  0.0000    11      8.5335       284.65  T2u          .    .  
 0.0000  0.0000  0.0000    12     13.9081       463.92  T1u          Y    .  
 0.0000  0.0000  0.0000    13     13.9081       463.92  T1u          Y    .  
 0.0000  0.0000  0.0000    14     13.9081       463.92  T1u          Y    .  
```

**Interpretation:**
- **T1u**: Triply degenerate (3D), IR-active, anti-symmetric with respect to inversion.
- **T2u**: Triply degenerate (3D), silent (neither IR nor Raman active).
- Note that the acoustic modes (bands 3-5 at 0 THz) and the unstable soft modes (bands 0-2) are all correctly identified.

### X Point $(0, 0.5, 0)$
To label the X point, we use the `irrep` backend and specify the `kpname`:
```bash
phonopy-irreps --params BaTiO3_phonopy_params.yaml --qpoint 0 0.5 0 --backend irrep --kpname X
```

**Raw CLI Output:**
```text
q-point: [0.0000, 0.5000, 0.0000]
Point group: Pm-3m

# qx      qy      qz      band  freq(THz)   freq(cm-1)   label
 0.0000  0.5000  0.0000     0     -4.8804      -162.79  X5+       
 0.0000  0.5000  0.0000     1     -4.8804      -162.79  X5+       
 0.0000  0.5000  0.0000     2      3.3171       110.65  X5-       
 0.0000  0.5000  0.0000     3      3.3171       110.65  X5-       
 0.0000  0.5000  0.0000     4      4.6550       155.27  X3-       
 0.0000  0.5000  0.0000     5      6.0282       201.08  X5+       
 0.0000  0.5000  0.0000     6      6.0282       201.08  X5+       
 0.0000  0.5000  0.0000     7      8.4478       281.79  X1+       
 0.0000  0.5000  0.0000     8      9.8687       329.19  X2+       
 0.0000  0.5000  0.0000     9     10.1483       338.51  X5-       
 0.0000  0.5000  0.0000    10     10.1483       338.51  X5-       
 0.0000  0.5000  0.0000    11     12.9610       432.33  X5+       
 0.0000  0.5000  0.0000    12     12.9610       432.33  X5+       
 0.0000  0.5000  0.0000    13     17.3048       577.23  X1+       
 0.0000  0.5000  0.0000    14     21.4201       714.50  X3-       
```

**Interpretation:**
- Labels like `X1+` and `X5-` follow the BCS standard.
- The `+`/`-` indicates **parity** (symmetric or anti-symmetric) with respect to inversion.
- Note that the IR and Raman activity columns are omitted for non-Gamma backends as activity rules are typically defined for the $\Gamma$ point.

### M Point $(0.5, 0.5, 0)$
```bash
phonopy-irreps --params BaTiO3_phonopy_params.yaml --qpoint 0.5 0.5 0 --backend irrep --kpname M
```

**Raw CLI Output:**
```text
q-point: [0.5000, 0.5000, 0.0000]
Point group: Pm-3m

# qx      qy      qz      band  freq(THz)   freq(cm-1)   label
 0.5000  0.5000  0.0000     0     -3.9928      -133.19  M2-       
 0.5000  0.5000  0.0000     1      3.0079       100.33  M5-       
 0.5000  0.5000  0.0000     2      3.0079       100.33  M5-       
 0.5000  0.5000  0.0000     3      3.3028       110.17  M3-       
 0.5000  0.5000  0.0000     4      5.9079       197.07  M2+       
 0.5000  0.5000  0.0000     5      8.4982       283.47  M5-       
 0.5000  0.5000  0.0000     6      8.4982       283.47  M5-       
 0.5000  0.5000  0.0000     7     10.4145       347.39  M2-       
 0.5000  0.5000  0.0000     8     10.5430       351.68  M3+       
 0.5000  0.5000  0.0000     9     10.5896       353.23  M5+       
 0.5000  0.5000  0.0000    10     10.5896       353.23  M5+       
 0.5000  0.5000  0.0000    11     13.3902       446.65  M5-       
 0.5000  0.5000  0.0000    12     13.3902       446.65  M5-       
 0.5000  0.5000  0.0000    13     13.9911       466.69  M1+       
 0.5000  0.5000  0.0000    14     22.0658       736.04  M4+       
```


### Tips for Success:
1. **Symmetry Precision**: If the space group is not detected correctly, try adjusting `--symprec` (e.g., `1e-3`).
2. **K-point Names**: Ensure the `--kpname` matches the label used in the BCS tables for that specific space group.
3. **Degeneracy**: If modes that should be degenerate have slightly different frequencies, increase `--degeneracy-tolerance`.
