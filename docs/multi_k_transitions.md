# Multi-$k$ Phase Transitions and Induced Representations

## 1. Background and Theory

In the Landau theory of continuous phase transitions, a structural distortion is described by an order parameter that transforms according to a single irreducible representation (irrep) of the high-symmetry parent space group $\mathcal{G}_0$.

When the critical wavevector $k$ lies at a generic point or at a high-symmetry point where the "star of $k$" contains multiple arms (i.e., applying all operations of $\mathcal{G}_0$ to $k$ generates a set of distinct wavevectors $\{k_1, k_2, \ldots, k_m\}$ not related by reciprocal lattice vectors), the full physical irreducible representation $\Gamma$ of $\mathcal{G}_0$ is constructed via **induction** from the "small representation" $\tilde{\Gamma}$ of the little group $\mathcal{G}_k$.

The little group $\mathcal{G}_k$ consists of all operations in $\mathcal{G}_0$ that leave $k$ invariant (modulo a reciprocal lattice vector). The small representation $\tilde{\Gamma}$ describes how the order parameter components associated with the specific wavevector $k_1$ transform among themselves. 

The dimension of the full physical representation $\Gamma$ is $m \times \dim(\tilde{\Gamma})$, where $m$ is the number of arms in the star of $k$.

### Limitation of `spgrep-modulation`

The `spgrep-modulation` package and its `IsotropyEnumerator` class are designed to operate exclusively on the **little group** and the **small representation**. This means it evaluates order parameter directions (OPDs) like `(a, 0)` or `(a, b)` strictly for a single wavevector $k$. 

It is structurally "blind" to multi-$k$ transitions, which correspond to structural modulations that are linear combinations of waves from different arms of the star. For example, if the star has 2 arms and the small representation is 2D, a valid physical transition might involve a 4D order parameter direction like `(a, 0, 0, a)`, which mixes components from $k_1$ and $k_2$. `spgrep-modulation` cannot enumerate isotropy subgroups for such OPDs.

## 2. Option B: Induced Full Representations

To discover these multi-$k$ transitions, we must manually build the full induced representation matrices for all operations in the space group and test higher-dimensional OPDs.

### Workflow

1.  **Determine the Star of $k$:**
    *   Start with the primary wavevector $k_1$.
    *   Apply the rotational part of every symmetry operation in the primitive parent space group $\mathcal{G}_0$ to $k_1$.
    *   Collect all unique wavevectors $k_i$ (modulo integer reciprocal lattice vectors). This forms the star $\{k_1, k_2, \ldots, k_m\}$.
    *   For each arm $k_i$, identify a "coset representative" operation $g_i \in \mathcal{G}_0$ such that $g_i k_1 = k_i$ (modulo a reciprocal lattice vector). Choose $g_1$ to be the identity operation.

2.  **Obtain the Small Representation:**
    *   Use `spgrep` to find the little group $\mathcal{G}_{k_1}$ and its small irreducible representations $\tilde{\Gamma}$.
    *   For each operation $h \in \mathcal{G}_{k_1}$, we have a matrix $\tilde{\Gamma}(h)$.

3.  **Build the Induced Representation Matrices:**
    *   For any operation $g \in \mathcal{G}_0$, its full representation matrix $D(g)$ is an $m \times m$ block matrix, where each block is of size $\dim(\tilde{\Gamma}) \times \dim(\tilde{\Gamma})$.
    *   To find the $(i, j)$-th block of $D(g)$, evaluate the operation $h_{ij} = g_i^{-1} g g_j$.
    *   If $h_{ij}$ belongs to the little group $\mathcal{G}_{k_1}$ (i.e., it leaves $k_1$ invariant), the block is given by:
        $D(g)_{i, j} = e^{-i 2\pi k_1 \cdot \mathbf{t}_{ij}} \tilde{\Gamma}(h_{ij})$
        where $\mathbf{t}_{ij}$ is the translational part of $h_{ij}$ *minus* the translation that brings $g_i^{-1} g g_j k_1$ strictly back to $k_1$ in the first Brillouin zone. Actually, the simpler standard induction formula is:
        $D(g)_{i, j} = \begin{cases} \tilde{\Gamma}(g_i^{-1} g g_j) & \text{if } g_i^{-1} g g_j \in \mathcal{G}_{k_1} \\ 0 & \text{otherwise} \end{cases}$
        (Care must be taken with the exact phase factors depending on the choice of basis functions and fractional translations, as discussed in representation theory of space groups).
        
    *   *Correction for physically real representation*: Since $spgrep$ often provides complex representations, we might need to physically real representations. Let's stick to the standard induction formula first.

4.  **Test Multi-$k$ Order Parameter Directions:**
    *   Define a set of high-symmetry directions in the $m \times \dim(\tilde{\Gamma})$ dimensional space (e.g., `(a, 0, a, 0)`, `(a, a, a, a)`, etc.).
    *   For each OPD $\mathbf{v}$, find the "isotropy subgroup" $\mathcal{I}_{\mathbf{v}} \subset \mathcal{G}_0$: the set of operations $g$ such that $D(g) \mathbf{v} = \mathbf{v}$.
    *   The operations in $\mathcal{I}_{\mathbf{v}}$ define the symmetry of the low-symmetry phase.

5.  **Identify Daughter Space Group:**
    *   Pass the surviving operations (rotations and translations in $\mathcal{I}_{\mathbf{v}}$) to the existing `_identify_daughter_spacegroup` function to determine the resulting space group number.

## 3. Implementation Specifics

*   **File:** `anaddb_irreps/anaddb_irreps/chiral_transitions.py`
*   **Functions to Add/Modify:**
    *   `_get_star_of_k(kpoint, rotations)`: Computes the star of $k$ and coset representatives.
    *   `_build_induced_representation(g_ops, k_star, coset_reps, little_group_ops, small_rep_matrices)`: Constructs the full block matrices $D(g)$.
    *   `_get_multi_k_opds(dim)`: Generates a list of candidate multi-dimensional OPDs.
    *   Update `_fallback_opd_search` (or create a new `_multi_k_search`) to use these full matrices to find the isotropy subgroups.

### Special Handling of Phases
Space group representations involve phase factors $e^{-i 2\pi \mathbf{k} \cdot \mathbf{t}}$. When implementing the induction $D(g)_{i, j} = \tilde{\Gamma}(g_i^{-1} g g_j)$, we must ensure that the small representation matrix $\tilde{\Gamma}$ correctly incorporates the fractional translations of the non-symmorphic operations, and the Bloch phase factors associated with moving between unit cells if $g_i^{-1} g g_j$ includes a lattice translation.

Specifically, if $g_i^{-1} g g_j = \{ R | \mathbf{v} \}$ where $R$ is in the little co-group of $k_1$, we can decompose it as $\{ R | \mathbf{v} \} = \{ E | \mathbf{t}_{lat} \} \{ R | \mathbf{v}_{frac} \}$ where $\{ R | \mathbf{v}_{frac} \}$ is the standard representative in the little group. Then $\tilde{\Gamma}(\{ R | \mathbf{v} \}) = e^{-i 2\pi k_1 \cdot \mathbf{t}_{lat}} \tilde{\Gamma}(\{ R | \mathbf{v}_{frac} \})$.
