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
```bash
phonopy-irreps --params BaTiO3_phonopy_params.yaml --qpoint 0 0 0
```
Expected result: Acoustic modes are $T_{1u}$, optical modes are $T_{1u}$ and $T_{2u}$.

### X Point $(0, 0.5, 0)$
To label the X point, we use the `irrep` backend and specify the `kpname`:
```bash
phonopy-irreps --params BaTiO3_phonopy_params.yaml --qpoint 0 0.5 0 --backend irrep --kpname X
```

**Results and Interpretation:**
- Labels like `X1+` and `X5-` follow the BCS standard.
- The `+`/`-` indicates **parity** (symmetric or anti-symmetric) with respect to inversion, as the little group for the $X$ point in this space group is centrosymmetric.
- `X5` is a **2D representation**, consistent with the doubly degenerate modes found at frequencies -4.88, 3.32, 6.03, 10.15, and 12.96 THz.

| Band Index | Freq (THz) | Irrep Label | Match Score |
|------------|------------|-------------|-------------|
| 0, 1       | -4.88      | X5+         | 1.00        |
| 2, 3       | 3.32       | X5-         | 1.00        |
| 4          | 4.66       | X3-         | 1.00        |
| 5, 6       | 6.03       | X5+         | 1.00        |
| 7          | 8.45       | X1+         | 1.00        |
| 8          | 9.87       | X2+         | 1.00        |
| 9, 10      | 10.15      | X5-         | 1.00        |
| 11, 12     | 12.96      | X5+         | 1.00        |
| 13         | 17.30      | X1+         | 1.00        |
| 14         | 21.42      | X3-         | 1.00        |

### M Point $(0.5, 0.5, 0)$
```bash
phonopy-irreps --params BaTiO3_phonopy_params.yaml --qpoint 0.5 0.5 0 --backend irrep --kpname M
```

**Results:**
| Band Index | Freq (THz) | Irrep Label | Match Score |
|------------|------------|-------------|-------------|
| 0          | -3.99      | M2-         | 1.00        |
| 1, 2       | 3.01       | M5-         | 1.00        |
| 3          | 3.30       | M3-         | 1.00        |
| 4          | 5.91       | M2+         | 1.00        |
| 5, 6       | 8.50       | M5-         | 1.00        |
| 7          | 10.41      | M2-         | 1.00        |
| 8          | 10.54      | M3+         | 1.00        |
| 9, 10      | 10.59      | M5+         | 1.00        |
| 11, 12     | 13.39      | M5-         | 1.00        |
| 13         | 13.99      | M1+         | 1.00        |
| 14         | 22.07      | M4+         | 1.00        |

### Tips for Success:
1. **Symmetry Precision**: If the space group is not detected correctly, try adjusting `--symprec` (e.g., `1e-3`).
2. **K-point Names**: Ensure the `--kpname` matches the label used in the BCS tables for that specific space group.
3. **Degeneracy**: If modes that should be degenerate have slightly different frequencies, increase `--degeneracy-tolerance`.
