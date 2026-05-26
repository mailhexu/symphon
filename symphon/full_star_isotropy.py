"""Full-star representation utilities for BCS-labeled irreps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib


@dataclass
class BcsFullStarContext:
    """Induced full-star representation context in the BCS cell setting.

    The context uses BCS-frame operations from ``irrep.SpaceGroupIrreps`` while
    testing k-vector equivalence modulo the primitive reciprocal lattice.  This
    matches BCS full-star OPDs such as SG 142 X ``(a, 0, a, 0)`` where the BCS
    conventional star has fewer arms than the raw primitive star.
    """

    qpoint_bcs: np.ndarray
    rotations_bcs: np.ndarray
    translations_bcs: np.ndarray
    rotations_prim: np.ndarray
    translations_prim: np.ndarray
    primitive_from_bcs: np.ndarray
    little_group_indices: list[int]
    little_rep_matrices: np.ndarray
    symprec: float = 1e-5

    @classmethod
    def from_spacegroup_irrep(
        cls,
        sg,
        qpoint_prim: np.ndarray,
        little_group_indices: list[int],
        little_rep_matrices: np.ndarray,
        symprec: float = 1e-5,
    ) -> "BcsFullStarContext":
        ref_uc = np.array(sg.refUC, dtype=float)
        ref_uc_inv = np.linalg.inv(ref_uc)
        shift_uc = np.array(sg.shiftUC, dtype=float)

        rotations = []
        translations = []
        rotations_prim = []
        translations_prim = []
        for sym in sg.symmetries:
            rot_prim = np.array(sym.rotation, dtype=int)
            trans_prim = np.array(sym.translation, dtype=float)
            rotations_prim.append(rot_prim)
            translations_prim.append(trans_prim)
            rotations.append(np.round(ref_uc_inv @ rot_prim @ ref_uc).astype(int))
            translations.append(
                (ref_uc_inv @ (trans_prim + rot_prim @ shift_uc - shift_uc)) % 1.0
            )

        return cls(
            qpoint_bcs=ref_uc.T @ np.array(qpoint_prim, dtype=float),
            rotations_bcs=np.array(rotations, dtype=int),
            translations_bcs=np.array(translations, dtype=float),
            rotations_prim=np.array(rotations_prim, dtype=int),
            translations_prim=np.array(translations_prim, dtype=float),
            primitive_from_bcs=ref_uc,
            little_group_indices=[int(i) for i in little_group_indices],
            little_rep_matrices=np.array(little_rep_matrices, dtype=complex),
            symprec=symprec,
        )

    @property
    def dim_small(self) -> int:
        return int(self.little_rep_matrices.shape[1])

    @property
    def primitive_reciprocal_from_bcs(self) -> np.ndarray:
        return self.primitive_from_bcs.T

    def _wrap_primitive_direct(self, vec_bcs: np.ndarray) -> np.ndarray:
        """Return primitive direct coordinates for a BCS-coordinate vector."""
        return self.primitive_from_bcs @ vec_bcs

    def _direct_equiv(self, a_bcs: np.ndarray, b_bcs: np.ndarray) -> tuple[bool, np.ndarray]:
        diff_bcs = np.array(a_bcs, dtype=float) - np.array(b_bcs, dtype=float)
        diff_prim = self._wrap_primitive_direct(diff_bcs)
        rounded = np.round(diff_prim)
        return bool(np.allclose(diff_prim - rounded, 0, atol=self.symprec)), rounded.astype(int)

    def _reciprocal_equiv(self, a_bcs: np.ndarray, b_bcs: np.ndarray) -> tuple[bool, np.ndarray]:
        diff_bcs = np.array(a_bcs, dtype=float) - np.array(b_bcs, dtype=float)
        diff_prim_recip = np.linalg.inv(self.primitive_reciprocal_from_bcs) @ diff_bcs
        rounded = np.round(diff_prim_recip)
        return bool(np.allclose(diff_prim_recip - rounded, 0, atol=self.symprec)), rounded.astype(int)

    def star(self) -> tuple[list[np.ndarray], list[int]]:
        arms: list[np.ndarray] = []
        reps: list[int] = []
        for i, rot in enumerate(self.rotations_bcs):
            candidate = rot @ self.qpoint_bcs
            if not any(self._reciprocal_equiv(candidate, arm)[0] for arm in arms):
                arms.append(candidate)
                reps.append(i)
        return arms, reps

    def _inverse_operation(self, rot: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rot_inv = np.linalg.inv(rot).astype(int)
        return rot_inv, -rot_inv @ trans

    def _find_little_group_op(
        self,
        rot: np.ndarray,
        trans: np.ndarray,
        left_idx: int | None = None,
        right_idx: int | None = None,
    ) -> tuple[int | None, np.ndarray | None]:
        for pos, idx in enumerate(self.little_group_indices):
            if not np.array_equal(rot, self.rotations_bcs[idx]):
                continue
            ok, lattice_shift_prim = self._direct_equiv(trans, self.translations_bcs[idx])
            if ok:
                if left_idx is not None and right_idx is not None:
                    raw_shift_prim = (
                        self.rotations_prim[left_idx] @ self.translations_prim[right_idx]
                        + self.translations_prim[left_idx]
                        - self.translations_prim[idx]
                    )
                    if np.allclose(raw_shift_prim - np.round(raw_shift_prim), 0, atol=self.symprec):
                        lattice_shift_prim = np.round(raw_shift_prim).astype(int)
                return pos, lattice_shift_prim
        return None, None

    def induced_matrices(self) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
        arms, reps = self.star()
        n_arms = len(arms)
        d = self.dim_small
        full = np.zeros((len(self.rotations_bcs), n_arms * d, n_arms * d), dtype=complex)

        for g_idx, (rot_g, trans_g) in enumerate(zip(self.rotations_bcs, self.translations_bcs)):
            for i_arm, rep_i in enumerate(reps):
                rot_i = self.rotations_bcs[rep_i]
                trans_i = self.translations_bcs[rep_i]
                rot_i_inv, trans_i_inv = self._inverse_operation(rot_i, trans_i)

                for j_arm, rep_j in enumerate(reps):
                    rot_j = self.rotations_bcs[rep_j]
                    trans_j = self.translations_bcs[rep_j]

                    rot_h = rot_i_inv @ (rot_g @ rot_j)
                    trans_h = rot_i_inv @ (rot_g @ trans_j + trans_g) + trans_i_inv

                    little_pos, lattice_shift_prim = self._find_little_group_op(rot_h, trans_h)
                    if little_pos is None or lattice_shift_prim is None:
                        continue

                    matched_idx = self.little_group_indices[little_pos]
                    rot_i_prim = self.rotations_prim[rep_i]
                    trans_i_prim = self.translations_prim[rep_i]
                    rot_i_prim_inv, trans_i_prim_inv = self._inverse_operation(rot_i_prim, trans_i_prim)
                    rot_g_prim = self.rotations_prim[g_idx]
                    trans_g_prim = self.translations_prim[g_idx]
                    rot_j_prim = self.rotations_prim[rep_j]
                    trans_j_prim = self.translations_prim[rep_j]
                    trans_h_prim = rot_i_prim_inv @ (rot_g_prim @ trans_j_prim + trans_g_prim) + trans_i_prim_inv
                    raw_shift_prim = trans_h_prim - self.translations_prim[matched_idx]
                    if np.allclose(raw_shift_prim - np.round(raw_shift_prim), 0, atol=self.symprec):
                        lattice_shift_prim = np.round(raw_shift_prim).astype(int)

                    qpoint_prim = np.linalg.inv(self.primitive_from_bcs.T) @ self.qpoint_bcs
                    phase = np.exp(-2j * np.pi * np.dot(qpoint_prim, lattice_shift_prim))
                    row = slice(i_arm * d, (i_arm + 1) * d)
                    col = slice(j_arm * d, (j_arm + 1) * d)
                    full[g_idx, row, col] = self.little_rep_matrices[little_pos] * phase

        return full, arms, reps

    def multiplication_table(self) -> np.ndarray:
        n = len(self.rotations_bcs)
        table = np.full((n, n), -1, dtype=int)
        for i in range(n):
            for j in range(n):
                rot = self.rotations_bcs[i] @ self.rotations_bcs[j]
                trans = self.rotations_bcs[i] @ self.translations_bcs[j] + self.translations_bcs[i]
                for k in range(n):
                    if not np.array_equal(rot, self.rotations_bcs[k]):
                        continue
                    if self._direct_equiv(trans, self.translations_bcs[k])[0]:
                        table[i, j] = k
                        break
        return table

    def closure_error(self, matrices: np.ndarray | None = None) -> float:
        if matrices is None:
            matrices = self.induced_matrices()[0]
        table = self.multiplication_table()
        worst = 0.0
        for i in range(len(matrices)):
            for j in range(len(matrices)):
                k = table[i, j]
                if k < 0:
                    return float("inf")
                err = np.linalg.norm(matrices[i] @ matrices[j] - matrices[k])
                worst = max(worst, float(err))
        return worst

    def _supercell_matrix_for_star(self, arms_bcs: list[np.ndarray], max_denom: int = 12) -> np.ndarray:
        denoms = [1, 1, 1]
        for arm_bcs in arms_bcs:
            arm_prim = np.linalg.inv(self.primitive_reciprocal_from_bcs) @ arm_bcs
            for i, value in enumerate(arm_prim):
                if np.isclose(value, round(value), atol=self.symprec):
                    continue
                for denom in range(1, max_denom + 1):
                    if np.isclose((value * denom) % 1.0, 0, atol=self.symprec):
                        import math

                        denoms[i] = abs(denoms[i] * denom) // math.gcd(denoms[i], denom)
                        break
        return np.diag(denoms)

    def identify_full_star_daughter(
        self,
        opd: np.ndarray,
        primitive_lattice: np.ndarray,
        matrices: np.ndarray | None = None,
        arms: list[np.ndarray] | None = None,
    ) -> tuple[int, str, int]:
        """Identify the daughter SG stabilizing a full-star OPD.

        Parameters
        ----------
        opd
            Full-star OPD vector ordered by star arm, then small-rep component.
        primitive_lattice
            Parent primitive lattice used with ``rotations_prim`` and
            ``translations_prim``.

        Returns
        -------
        (number, symbol, operation_count)
        """
        if matrices is None or arms is None:
            matrices, arms, _ = self.induced_matrices()

        opd = np.array(opd, dtype=complex)
        supercell_matrix = self._supercell_matrix_for_star(arms)
        supercell_inv = np.linalg.inv(supercell_matrix)
        lattice = supercell_matrix @ np.array(primitive_lattice, dtype=float)
        dim_small = self.dim_small

        lattice_translations = []
        from itertools import product

        bound = int(np.max(np.diag(supercell_matrix))) + 2
        for n_tuple in product(range(-bound, bound + 1), repeat=3):
            n = np.array(n_tuple, dtype=int)
            n_super = supercell_inv @ n
            if np.all(n_super > -self.symprec) and np.all(n_super < 1 - self.symprec):
                lattice_translations.append(n)

        sc_rots = []
        sc_trans = []
        for rot, trans, mat_g in zip(self.rotations_prim, self.translations_prim, matrices):
            for lattice_translation in lattice_translations:
                phase_mat = np.zeros_like(mat_g)
                for arm_i, arm_bcs in enumerate(arms):
                    arm_prim = np.linalg.inv(self.primitive_reciprocal_from_bcs) @ arm_bcs
                    phase = np.exp(-2j * np.pi * np.dot(arm_prim, lattice_translation))
                    start = arm_i * dim_small
                    stop = start + dim_small
                    phase_mat[start:stop, start:stop] = np.eye(dim_small) * phase

                if np.linalg.norm((phase_mat @ mat_g) @ opd - opd) >= 1e-5:
                    continue

                rot_super = supercell_inv @ rot @ supercell_matrix
                trans_super = (supercell_inv @ (trans + lattice_translation)) % 1.0
                if any(
                    np.allclose(rot_super, existing_rot, atol=self.symprec)
                    and np.allclose(trans_super, existing_trans, atol=self.symprec)
                    for existing_rot, existing_trans in zip(sc_rots, sc_trans)
                ):
                    continue
                sc_rots.append(rot_super)
                sc_trans.append(trans_super)

        if not sc_rots:
            return 0, "Unknown", 0

        sg_type = spglib.get_spacegroup_type_from_symmetry(
            np.round(sc_rots).astype("intc"),
            np.array(sc_trans, dtype="double"),
            lattice=lattice,
            symprec=self.symprec,
        )
        if sg_type is None:
            return 0, "Unknown", len(sc_rots)

        number = sg_type.get("number", 0) if isinstance(sg_type, dict) else getattr(sg_type, "number", 0)
        symbol = (
            sg_type.get("international_short", "")
            if isinstance(sg_type, dict)
            else getattr(sg_type, "international_short", "")
        )
        return int(number), str(symbol), len(sc_rots)
