# Methodology for Identification of IR and Raman Activity

In the harmonic approximation, the vibrational modes of a crystal at the Brillouin zone center ($\Gamma$ point) can be classified according to the irreducible representations (irreps) of the crystal's point group $P$. This document describes the group-theoretical method implemented in `anaddb_irreps` to determine the spectroscopic activity of these modes.

## 1. Theoretical Framework

The total representation of the phonon modes, $\Gamma_{ph}$, is a reducible representation of the point group $P$. It can be decomposed into a direct sum of irreps $\Gamma_i$:
$$\Gamma_{ph} = \sum_i n_i \Gamma_i$$
where $n_i$ is the multiplicity of the $i$-th irrep. Spectroscopic activity is determined by the transformation properties of the physical operators responsible for the transition.

### 1.1 Infrared Activity
A phonon mode is infrared (IR) active if its corresponding irrep $\Gamma_i$ transforms like a component of the electric dipole moment vector $\mathbf{p}$. In group theory terms, $\Gamma_i$ must be contained in the **polar vector representation** $\Gamma_{V}$. 

The character of the vector representation for a symmetry operation $R$ is given by the trace of its $3 \times 3$ Cartesian rotation matrix $\mathbf{R}$:
$$\chi_{V}(R) = \text{Tr}(\mathbf{R})$$

### 1.2 Raman Activity
A phonon mode is Raman active if its corresponding irrep $\Gamma_i$ transforms like a component of the symmetric polarizability tensor $\boldsymbol{\alpha}$. This corresponds to the **symmetric square** of the vector representation, $[\Gamma_{V}]^2$.

The character of this symmetric tensor representation is calculated using the following relation:
$$\chi_{\alpha}(R) = \frac{1}{2} \left( [\chi_{V}(R)]^2 + \chi_{V}(R^2) \right)$$

## 2. Numerical Identification

To determine if a specific irrep $\Gamma_i$ (identified by its character $\chi_i$) is active, we calculate its multiplicity $n$ within the respective reducible representation ($\Gamma_{V}$ or $\Gamma_{\alpha}$) using the standard orthogonality relation:
$$n_i = \frac{1}{g} \sum_{R \in P} \chi_i(R)^* \chi_{red}(R)$$
where:
- $g$ is the order of the point group (total number of operations).
- $\chi_{red}$ is either $\chi_{V}$ (for IR) or $\chi_{\alpha}$ (for Raman).

If $n_i > 0$, the mode is classified as active. In standard crystals, $n_i$ is typically 1 for active irreps and 0 for silent ones.

---

## 3. Case Study: TmFeO3

TmFeO3 in its orthorhombic phase belongs to the space group $Pnma$ (No. 62). At the $\Gamma$ point, the relevant symmetry is the point group $mmm$ ($D_{2h}$).

### 3.1 Selection Rules for Point Group $mmm$
The point group $mmm$ contains 8 irreps. Using the formulas above, the activity is identified as follows:

| Irrep | $\Gamma_{V}$ (IR) | $[\Gamma_{V}]^2$ (Raman) | Activity |
|-------|:-----------------:|:------------------------:|:---------|
| $A_g$   | 0                 | 1                        | Raman    |
| $B_{1g}$ | 0                 | 1                        | Raman    |
| $B_{2g}$ | 0                 | 1                        | Raman    |
| $B_{3g}$ | 0                 | 1                        | Raman    |
| $A_u$   | 0                 | 0                        | Silent   |
| $B_{1u}$ | 1                 | 0                        | IR       |
| $B_{2u}$ | 1                 | 0                        | IR       |
| $B_{3u}$ | 1                 | 0                        | IR       |

### 3.2 Calculated Results
Running `anaddb_irreps` on TmFeO3 yields 60 modes (20 atoms in the primitive cell). The results perfectly match the group-theoretical predictions:

```text
# qx      qy      qz      band  freq(THz)   freq(cm-1)   label        IR  Raman
 0.0000  0.0000  0.0000     3      2.1839        72.85  Au           .    .  
 0.0000  0.0000  0.0000     4      3.2081       107.01  B3u          Y    .  
 0.0000  0.0000  0.0000     5      3.2489       108.37  B2g          .    Y  
 0.0000  0.0000  0.0000     6      3.4713       115.79  Ag           .    Y  
```

- **Bands 0-2**: Acoustic modes ($B_{1u} + B_{2u} + B_{3u}$), IR active.
- **Band 3**: $A_u$ mode at 2.18 THz, correctly identified as **silent** (both IR and Raman columns are empty `.`).
- **Band 5**: $B_{2g}$ mode at 3.25 THz, correctly identified as **Raman active**.
- **Band 59**: $A_g$ mode at 20.32 THz, correctly identified as **Raman active**.

The implementation successfully handles the **Mutual Exclusion Principle** for this centrosymmetric crystal: no mode is both IR and Raman active.
