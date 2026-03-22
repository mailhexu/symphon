# Example: Chiral Phase Transition in Space Group 109 ($I4_1md$)

Here is a detailed, step-by-step breakdown of how the `symphon` code analyzes Space Group 109 ($I4_1md$) to find a Class II chiral phase transition, identifying the Order Parameter Direction (OPD) and computing the daughter space group.

This trace specifically follows the execution inside `symphon/chiral/transitions.py` for the transition occurring at the **X q-point** with the **2X1 irreducible representation (irrep)**.

### Step 1: Initialization & Parent Group Analysis
The code starts by initializing the `ChiralTransitionFinder(109)`.
* **Action:** `spglib` is called to generate the space group information for SG 109.
* **Intermediate Result:** The parent is identified as SG 109 ($I4_1md$). The code confirms it is achiral (Class I) because it contains improper operations (specifically, mirrors and glides). For example, it extracts parent improper operations like:
  * Mirror: $R = \begin{bmatrix}-1 & 0 & 0\\ 0 & 1 & 0\\ 0 & 0 & 1\end{bmatrix}$, $T = [0, 0, 0]$
  * Glide: $R = \begin{bmatrix}0 & 1 & 0\\ 1 & 0 & 0\\ 0 & 0 & 1\end{bmatrix}$, $T = [0, 0.5, 0.25]$

### Step 2: Q-point and Irrep Selection
The `find_chiral_transitions()` method requests the special q-points for the lattice and the irreps at each q-point.
* **Action:** It calls `get_special_qpoints()` and evaluates each one. We will follow the loop iteration where the q-point is the zone boundary point **X**.
* **Intermediate Result:** The q-point `X` has the fractional coordinates `[0.5, 0.5, 0.0]`. 
* **Action:** It then calls `get_irreps_at_qpoint()` using `spgrep` to compute the representations. It identifies the 2-dimensional irrep labeled **2X1**.

### Step 3: Enumerating Isotropy Subgroups and Generating the OPD
The code needs to find how the symmetry breaks for this specific irrep.
* **Action:** It calls `enumerate_isotropy_subgroups()` and passes the small representation matrices (`small_rep`), rotations, and translations of the little group for `2X1`.
* **Under the hood:** This function delegates to `IsotropyEnumerator` from the `spgrep-modulation` library to enumerate the maximal isotropy subgroups.
* **Intermediate Result:** The enumerator outputs a set of numerical vectors (OPDs) that break the symmetry. One of the derived numerical OPDs is:
  * **OPD (Numerical):** `[1.0, 0.0, 1.0, 0.0]`
* **Action:** This numerical vector is passed to the internal helper `opd_to_symbolic()`, which formats it into a human-readable string.
  * **OPD (Symbolic):** `(a,0,a,0)`

### Step 4: Constructing the Daughter Space Group 
Now the code must determine what space group results when the crystal is distorted along the `(a,0,a,0)` direction.
* **Action:** It calls `_identify_daughter_spacegroup()`. 
* Since the `X` q-point `[0.5, 0.5, 0.0]` has fractional denominators of 2, the code calculates that a $2 \times 2 \times 1$ commensurate **supercell** is required (`S_conv = diag(2,2,1)`).
* **Filtering Operations:** For every original symmetry operation $j$, the code checks if the OPD remains invariant under the representation matrix multiplied by the phase shift of the supercell translation $n$:
  $e^{-2\pi i (q \cdot n)} \cdot M_j \cdot \text{OPD} = \text{OPD}$
  (where $M_j$ is the small representation matrix, and the difference norm must be `< 1e-5`).
* **Action:** Operations that satisfy this condition are transformed into the supercell basis and appended to surviving `sc_rots` (rotations) and `sc_trans` (translations).
* **Identification:** These surviving supercell operations are fed into `spglib.get_spacegroup_type_from_symmetry()`.
* **Intermediate Result:** `spglib` identifies the surviving symmetry operations as **Space Group 76 ($P4_1$)**.

### Step 5: Classifying as Class II Sohncke Group
With the daughter space group identified, the code applies the chiral classification labels.
* **Action:** It checks if SG 76 is chiral using `is_sohncke(76)`. Since SG 76 only contains proper rotations and translations, the check returns `True`.
* **Action:** The code then calls `get_sohncke_class(76)`. This function checks if SG 76 is in the hardcoded list `_ENANTIOMORPHOUS_PAIRS` (11 specific pairs of space groups). 
* **Intermediate Result:** SG 76 ($P4_1$) forms an enantiomorphous pair with SG 78 ($P4_3$). Therefore, the function returns `SohnckeClass.CLASS_II`.

### Step 6: Computing Lost Operations and Enantiomeric Domains
Finally, the transition metadata is finalized by comparing the parent and daughter groups.
* **Action:** The method `_analyze_lost_operations()` subtracts the daughter's surviving operations from the parent's original operations.
* **Intermediate Result:** The code logs that several improper operations were lost, for example:
  * Lost mirror: $R = \begin{bmatrix}1 & 0 & 0\\ 1 & 0 & -1\\ 1 & -1 & 0\end{bmatrix}$
  * Lost glide: $R = \begin{bmatrix}0 & 1 & 0\\ 1 & 0 & 0\\ 0 & 0 & 1\end{bmatrix}$
* **Action:** Because improper operations (mirrors/glides) were lost, `_count_enantiomeric_domains()` calculates that the spontaneous symmetry breaking creates left-handed and right-handed variants.
* **Final Output:** The transition is assigned `enantiomeric_domain_count = 2`, completing the entry for SG 109 $\rightarrow$ SG 76 at the X-point.