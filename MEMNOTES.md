# Project Memory & Key Concepts (Memnotes)

## Chiral Phase Transitions: Conceptual Framework

When analyzing materials for chirality, we systematically categorize the symmetry-breaking pathways based on two independent order parameters: **Structural/Displacive** (time-even, usually driven by phonons) and **Magnetic** (time-odd, spin ordering).

A strictly **chiral** state requires that its space group (or magnetic space group) contains **only operations where $\det(R) = +1$**. 
* If an operation has $\det(R) = -1$ (like $m$, $\bar{1}$), it is geometrically achiral.
* If an operation has $\det(R) = -1$ coupled with time-reversal (like $m'$, $\bar{1}'$), it is *magnetically* achiral.

### 1. Purely Structural Chiral Transitions
* **Pathway:** Achiral Paramagnetic $\to$ Chiral Paramagnetic
* **Implementation:** Use `ChiralTransitionFinder`. Finds structural phonons that break all improper spatial symmetries. Time-reversal is unbroken.

### 2. Purely Magnetic Chiral Transitions
* **Pathway:** Achiral Paramagnetic $\to$ Achiral Lattice + Chiral Spin Order
* **Implementation:** Use `MagneticTransitionFinder` or `AbstractMagneticTransitionFinder`. 
* **Note:** Very rare in simple collinear systems. Breaking $\bar{1}$ magnetically often preserves $\bar{1}'$ (time-reversed inversion), which remains macroscopically achiral. True magnetic chirality usually requires non-collinear, non-coplanar, or multi-$\mathbf{k}$ magnetic structures to destroy *both* $\bar{1}$ and $\bar{1}'$.

### 3. Sequential Magneto-Structural Transitions
* **Pathway 3A (Structure leads):** Achiral Paramagnetic $\xrightarrow{T_1}$ Chiral Paramagnetic $\xrightarrow{T_2}$ Chiral Magnetic. 
  * The lattice distorts into a chiral space group first. Subsequent magnetic ordering is naturally forced into a chiral magnetic space group.
* **Pathway 3B (Magnetism leads):** Achiral Paramagnetic $\xrightarrow{T_1}$ Achiral Magnetic $\xrightarrow{T_2}$ Chiral Magnetic.
  * System orders magnetically first but retains a symmetry like $\bar{1}'$. A secondary structural distortion later breaks the remaining $\bar{1}'$ symmetry.
  * **Workflow:** Find the achiral magnetic transition first, then use structural irrep analysis on that intermediate BNS group as the new parent phase.

### 4. Simultaneous / Improper Coupled Transitions
* **Pathway:** Achiral Paramagnetic $\to$ Chiral Magnetic
* **Mechanism:** Structural and magnetic distortions condense at the same temperature, often driven by strong free-energy coupling (e.g., $M^2 P$). The final state is simultaneously structurally and magnetically chiral.


### 5. The "Intersection Mechanism" (Achiral + Achiral = Chiral)
The combined symmetry of a system is the mathematical intersection of the structural and magnetic subgroups ($G_{\text{final}} = G_{\text{struct}} \cap G_{\text{mag}}$). 
Because chirality requires the **total absence** of improper operations:
* If $G_{\text{struct}}$ is Achiral (preserves $m_x$)
* If $G_{\text{mag}}$ is Achiral (preserves $\bar{1}'$)
* The intersection $G_{\text{final}}$ preserves neither. The final state is **Chiral**.
This provides a highly robust "hidden" pathway to design chiral materials by intentionally mismatching the symmetry-breaking directions of magnetic and structural order parameters.
