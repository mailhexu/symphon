# Project Memory & Key Concepts (Memnotes)

## Chiral Phase Transitions: Conceptual Framework

When analyzing materials for chirality, we systematically categorize the symmetry-breaking pathways based on two independent order parameters: **Structural/Displacive** (time-even, usually driven by phonons) and **Magnetic** (time-odd, spin ordering).

A strictly **chiral** state requires that its space group (or magnetic space group) contains **only operations where $\det(R) = +1$**. 
* If an operation has $\det(R) = -1$ (like $m$, $\bar{1}$), it is geometrically achiral.
* If an operation has $\det(R) = -1$ coupled with time-reversal (like $m'$, $\bar{1}'$), it is *magnetically* achiral.

### The Chirality Intersection Table
The final symmetry of a material is the intersection of its structural and magnetic symmetries ($G_{\text{final}} = G_{\text{struct}} \cap G_{\text{mag}}$). 

| Structural Symmetry ($G_{\text{struct}}$) | Magnetic Symmetry ($G_{\text{mag}}$) | Combined Symmetry ($G_{\text{final}}$) | Physical Mechanism |
| :--- | :--- | :--- | :--- |
| **Chiral** | Achiral | **Chiral** | The chiral lattice breaks all improper symmetries. Magnetism cannot restore them. |
| Achiral | **Chiral** | **Chiral** | The complex spin texture breaks all improper symmetries of the symmetric lattice. |
| **Chiral** | **Chiral** | **Chiral** | Both independently break all improper symmetries. |
| Achiral | Achiral | Achiral | Both preserve at least one common improper symmetry (e.g., both preserve $\bar{1}$). |
| Achiral | Achiral | **Chiral** | **Intersection Mechanism:** $G_{\text{struct}}$ preserves one improper symmetry (e.g., $m_x$), $G_{\text{mag}}$ preserves a *different* one (e.g., $\bar{1}'$). Their intersection leaves no shared improper symmetries! |

---

## The 4 Categories of Transitions & How to Find the MSG

### Category 1: Purely Structural Chiral Transitions
* **Pathway:** Achiral Paramagnetic $\to$ Chiral Paramagnetic
* **Mechanism:** Structural phonons break all improper spatial symmetries. Time-reversal is unbroken.
* **How to find the MSG (With Structure):**
  1. Use `ChiralTransitionFinder` to find the purely structural subgroup (a Sohncke group).
  2. Because time-reversal ($\theta$) is unbroken, the resulting MSG is just the **Type II (grey) magnetic space group** of the resulting structural space group. For example, if the lattice distorts to $P4_1$, the MSG is $P4_11'$ (BNS number ends in `.2`).
* **Structure-Free (Abstract) Workflow:** Exact same as above. `ChiralTransitionFinder(spg_number)` is structure-free.

### Category 2: Purely Magnetic Chiral Transitions
* **Pathway:** Achiral Paramagnetic $\to$ Achiral Lattice + Chiral Spin Order
* **Mechanism:** Very rare. True magnetic chirality from a symmetric lattice requires non-collinear or multi-$\mathbf{k}$ magnetic structures to destroy *both* $\bar{1}$ and $\bar{1}'$.
* **How to find the MSG (With Structure):**
  1. Use `MagneticTransitionFinder(cell, mag_sites)`. 
  2. Filter the output for `is_chiral == True`. This directly yields the BNS number and the Order Parameter Direction (OPD) of the physically realisable chiral spin texture.
* **Structure-Free (Abstract) Workflow:** Use `AbstractMagneticTransitionFinder(spg_number)` and filter for `is_chiral == True` to evaluate generic time-odd order parameters.

### Category 3: Sequential Magneto-Structural Transitions
Achieved in two distinct thermodynamic steps at different temperatures ($T_1$ and $T_2$).

* **Pathway 3A (Structure leads):** Achiral Paramag $\xrightarrow{T_1}$ Chiral Paramag $\xrightarrow{T_2}$ Chiral Magnetic. 
  * **How to find the MSG (With Structure):**
    1. First, use `ChiralTransitionFinder` to identify the intermediate structurally chiral phase (e.g., $P4_1$).
    2. Construct the actual atomic geometry of this intermediate phase.
    3. Pass this new, relaxed, lower-symmetry cell into `MagneticTransitionFinder`. Any resulting magnetic order will naturally have a chiral MSG because the parent lattice is already chiral.
  * **Structure-Free (Abstract) Workflow:** Extract the intermediate SG number from `ChiralTransitionFinder` and feed it into `AbstractMagneticTransitionFinder`.

* **Pathway 3B (Magnetism leads / Intersection Mechanism):** Achiral Paramag $\xrightarrow{T_1}$ Achiral Mag $\xrightarrow{T_2}$ Chiral Mag.
  * **How to find the MSG (With Structure):**
    1. Use `MagneticTransitionFinder` on the parent phase. Locate an *achiral* MSG (e.g., one that retains $\bar{1}'$).
    2. To find the $T_2$ transition, treat this achiral MSG as your new parent phase. 
    3. Run a structural symmetry-breaking analysis using the operations of the achiral MSG to see which structural distortions (phonons) break the remaining $\bar{1}'$ symmetry. The intersection of these two phases defines the final chiral MSG.
  * **Structure-Free (Abstract) Workflow:** Evaluate time-even representations of the abstract intermediate MSG to identify abstract isotropy subgroups that break the remaining improper operations.

### Category 4: Simultaneous / Improper Coupled Transitions
* **Pathway:** Achiral Paramagnetic $\to$ Chiral Magnetic
* **Mechanism:** Structural and magnetic distortions condense at the same temperature due to strong free-energy coupling (e.g., $M^2 P$). 
* **How to find the MSG (With Structure):**
  1. Calculate the basis vectors for both the structural distortion and the magnetic order.
  2. Use `spglib.get_magnetic_symmetry_dataset()` on a custom supercell that includes *both* the atomic displacements and the spin vectors simultaneously.
* **Structure-Free (Abstract) Workflow:** Pure group theory requires evaluating the isotropy subgroups of a *coupled* representation (tensor product of time-even structural and time-odd magnetic irreps).
