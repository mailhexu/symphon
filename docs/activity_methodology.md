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

## 2. Computational Implementation Steps

The `anaddb_irreps` package automates the spectroscopic identification through the following algorithmic steps:

1.  **Symmetry Analysis**: The tool identifies the point group $P$ of the crystal structure at the $\Gamma$ point using `spglib`. For `TmFeO3`, this is the centrosymmetric orthorhombic group $mmm$ ($D_{2h}$).
2.  **Character Table Retrieval**: The character table for the identified point group is retrieved from the internal database (via `phonopy`). This table contains the characters $\chi_i(R)$ for each irreducible representation $\Gamma_i$.
3.  **Reducible Representation Construction**:
    *   The **IR-active** reducible representation $\Gamma_{V}$ is constructed by calculating the trace of each symmetry operation's $3 \times 3$ Cartesian rotation matrix: $\chi_{V}(R) = \text{Tr}(\mathbf{R})$.
    *   The **Raman-active** reducible representation $\Gamma_{\alpha}$ is constructed using the symmetric square formula: $\chi_{\alpha}(R) = \frac{1}{2} \left( [\chi_{V}(R)]^2 + \chi_{V}(R^2) \right)$. This correctly accounts for the transformation of symmetric tensors like the polarizability tensor.
4.  **Multiplicity Projection**: For every irrep $\Gamma_i$ in the point group, the tool calculates its multiplicity $n_i^{IR}$ and $n_i^{Raman}$ within the active representations using the master formula:
    $$n_i = \frac{1}{Order(P)} \sum_{R \in P} \chi_i(R)^* \chi_{red}(R)$$
5.  **Spectroscopic Tagging**: 
    *   An irrep is tagged as **IR-active** (`IR = Y`) if $n_i^{IR} \geq 1$.
    *   An irrep is tagged as **Raman-active** (`Raman = Y`) if $n_i^{Raman} \geq 1$.
    *   Phonon modes are then assigned these tags based on their identified irrep label.

---

## 3. Case Study: TmFeO3 (Full Spectroscopic Analysis)

TmFeO3 ($Pnma$, No. 62) has 20 atoms in its primitive cell, leading to 60 phonon modes. At the $\Gamma$ point ($mmm$ symmetry), the implementation correctly yields the following complete spectroscopic table:

```text
q-point: [0.0000, 0.0000, 0.0000]
Point group: mmm

# qx      qy      qz      band  freq(THz)   freq(cm-1)   label        IR  Raman
 0.0000  0.0000  0.0000     0     -0.1231        -4.11  B1u          Y    .  
 0.0000  0.0000  0.0000     1     -0.1031        -3.44  B3u          Y    .  
 0.0000  0.0000  0.0000     2     -0.0680        -2.27  B2u          Y    .  
 0.0000  0.0000  0.0000     3      2.1839        72.85  Au           .    .  
 0.0000  0.0000  0.0000     4      3.2081       107.01  B3u          Y    .  
 0.0000  0.0000  0.0000     5      3.2489       108.37  B2g          .    Y  
 0.0000  0.0000  0.0000     6      3.4713       115.79  Ag           .    Y  
 0.0000  0.0000  0.0000     7      3.5300       117.75  B3g          .    Y  
 0.0000  0.0000  0.0000     8      3.6628       122.18  B1u          Y    .  
 0.0000  0.0000  0.0000     9      3.7218       124.15  B1g          .    Y  
 0.0000  0.0000  0.0000    10      3.9842       132.90  Ag           .    Y  
 0.0000  0.0000  0.0000    11      4.5035       150.22  B2u          Y    .  
 0.0000  0.0000  0.0000    12      4.8692       162.42  Au           .    .  
 0.0000  0.0000  0.0000    13      5.0228       167.54  B2g          .    Y  
 0.0000  0.0000  0.0000    14      5.1475       171.70  B2u          Y    .  
 0.0000  0.0000  0.0000    15      5.4390       181.42  B1u          Y    .  
 0.0000  0.0000  0.0000    16      6.0491       201.78  B3u          Y    .  
 0.0000  0.0000  0.0000    17      6.1103       203.82  Au           .    .  
 0.0000  0.0000  0.0000    18      6.8098       227.15  Au           .    .  
 0.0000  0.0000  0.0000    19      7.4116       247.22  B3u          Y    .  
 0.0000  0.0000  0.0000    20      7.4418       248.23  B1g          .    Y  
 0.0000  0.0000  0.0000    21      7.7644       258.99  B1u          Y    .  
 0.0000  0.0000  0.0000    22      7.9374       264.76  B2u          Y    .  
 0.0000  0.0000  0.0000    23      8.1797       272.85  Ag           .    Y  
 0.0000  0.0000  0.0000    24      8.7735       292.65  B1u          Y    .  
 0.0000  0.0000  0.0000    25      9.1619       305.61  B3u          Y    .  
 0.0000  0.0000  0.0000    26      9.2071       307.12  B3g          .    Y  
 0.0000  0.0000  0.0000    27      9.7484       325.17  Au           .    .  
 0.0000  0.0000  0.0000    28      9.8554       328.74  B2u          Y    .  
 0.0000  0.0000  0.0000    29     10.0316       334.62  B2g          .    Y  
 0.0000  0.0000  0.0000    30     10.2605       342.25  B1u          Y    .  
 0.0000  0.0000  0.0000    31     10.3350       344.74  B3u          Y    .  
 0.0000  0.0000  0.0000    32     10.4629       349.01  Ag           .    Y  
 0.0000  0.0000  0.0000    33     10.7026       357.00  B3u          Y    .  
 0.0000  0.0000  0.0000    34     10.7738       359.38  B2g          .    Y  
 0.0000  0.0000  0.0000    35     11.2588       375.55  B1u          Y    .  
 0.0000  0.0000  0.0000    36     11.8975       396.86  B1g          .    Y  
 0.0000  0.0000  0.0000    37     12.3225       411.04  B2u          Y    .  
 0.0000  0.0000  0.0000    38     12.5166       417.51  Au           .    .  
 0.0000  0.0000  0.0000    39     13.5718       452.71  Ag           .    Y  
 0.0000  0.0000  0.0000    40     13.6374       454.90  B1u          Y    .  
 0.0000  0.0000  0.0000    41     13.7093       457.29  B3g          .    Y  
 0.0000  0.0000  0.0000    42     14.1067       470.55  B1g          .    Y  
 0.0000  0.0000  0.0000    43     14.1747       472.82  B3u          Y    .  
 0.0000  0.0000  0.0000    44     14.2010       473.69  Ag           .    Y  
 0.0000  0.0000  0.0000    45     14.6417       488.39  B3g          .    Y  
 0.0000  0.0000  0.0000    46     15.4600       515.69  B2g          .    Y  
 0.0000  0.0000  0.0000    47     15.5114       517.40  Au           .    .  
 0.0000  0.0000  0.0000    48     15.7773       526.27  B2u          Y    .  
 0.0000  0.0000  0.0000    49     15.8631       529.14  Ag           .    Y  
 0.0000  0.0000  0.0000    50     16.5121       550.78  -            .    .  
 0.0000  0.0000  0.0000    51     16.5166       550.93  -            .    .  
 0.0000  0.0000  0.0000    52     16.5474       551.96  -            .    .  
 0.0000  0.0000  0.0000    53     16.9049       563.89  Au           .    .  
 0.0000  0.0000  0.0000    54     16.9069       563.95  B2g          .    Y  
 0.0000  0.0000  0.0000    55     16.9279       564.65  B3u          Y    .  
 0.0000  0.0000  0.0000    56     17.7106       590.76  B1u          Y    .  
 0.0000  0.0000  0.0000    57     18.3283       611.37  B1g          .    Y  
 0.0000  0.0000  0.0000    58     20.0517       668.85  B2g          .    Y  
 0.0000  0.0000  0.0000    59     20.3213       677.85  B3g          .    Y  
```

