"""
Phonon-induced chiral phase transitions.

This module contains the logic for finding phase transitions from a parent
space group to a chiral Sohncke daughter space group.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from pathlib import Path
import json
import numpy as np

import spglib
from irrep.spacegroup_irreps import SpaceGroupIrreps
try:
    from irreptables.irreps import IrrepTable
except ImportError:
    from irreptables import IrrepTable  # type: ignore

from spgrep import get_spacegroup_irreps, get_spacegroup_irreps_from_primitive_symmetry
try:
    from spgrep.rep.representation import get_character
except ImportError:
    from spgrep.representation import get_character

from spgrep_modulation.isotropy import IsotropyEnumerator

from .sohncke import (
    ImproperOperationType, 
    SohnckeClass, 
    is_sohncke, 
    get_sohncke_class, 
    get_enantiomorph_partner, 
    get_screw_notation,
    _CACHE_DIR,
    _get_cache_path,
    _save_transitions_cache,
    _load_transitions_cache
)
from .ops import (
    classify_improper_operation, 
    has_improper_operations, 
    get_operation_description, 
    rotation_to_jones
)

from symphon.symmetry_identification import get_lattice_translations_for_supercell

# Flags are kept for compatibility if needed, but are now always True
HAS_SPGLIB = True
HAS_IRREP = True
HAS_SPGREP = True
HAS_SPGREP_MODULATION = True # Used but not defined in original


def _spglib_attr(sg_type: Any, field: str, default: Any = None) -> Any:
    """Read a field from a spglib type object (dict or named-attribute object)."""
    if sg_type is None:
        return default
    if isinstance(sg_type, dict):
        return sg_type.get(field, default)
    return getattr(sg_type, field, default)

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LostOperation:
    """Information about a symmetry operation lost in a phase transition."""
    operation_type: ImproperOperationType
    rotation: np.ndarray
    translation: np.ndarray
    description: str
    jones_symbol: str


@dataclass
class OrderParameterDirection:
    """Order parameter direction in irrep space."""
    numerical: np.ndarray
    symbolic: str
    num_free_params: int


@dataclass
class SpaceGroupInfo:
    """Symmetry information for a space group."""
    number: int
    symbol: str
    rotations: np.ndarray
    translations: np.ndarray
    point_group_symbol: str
    order: int
    
    # Primitive setting info
    primitive_rotations: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))
    primitive_translations: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    primitive_lattice: np.ndarray = field(default_factory=lambda: np.eye(3))

    @property
    def sohncke_class(self) -> SohnckeClass:
        return get_sohncke_class(self.number)

    @property
    def is_sohncke(self) -> bool:
        return is_sohncke(self.number)

    def has_inversion(self) -> bool:
        """Check if space group contains inversion."""
        for rot in self.rotations:
            if np.allclose(rot, -np.eye(3)):
                return True
        return False

    def count_mirrors(self) -> int:
        """Count pure mirror operations."""
        count = 0
        for rot, trans in zip(self.rotations, self.translations):
            op_type = classify_improper_operation(rot, trans)
            if op_type == ImproperOperationType.MIRROR:
                count += 1
        return count

    def count_glides(self) -> int:
        """Count glide operations."""
        count = 0
        for rot, trans in zip(self.rotations, self.translations):
            op_type = classify_improper_operation(rot, trans)
            if op_type == ImproperOperationType.GLIDE:
                count += 1
        return count

    def get_operations_report(self) -> str:
        """Get a detailed report of symmetry operations in this space group."""
        lines = []
        lines.append(f"{'Operation':<10} {'Type':<15} {'Description':<25} {'Jones':<20}")
        lines.append("-" * 75)
        for i, (rot, trans) in enumerate(zip(self.rotations, self.translations), 1):
            op_type = classify_improper_operation(rot, trans)
            type_str = op_type.value if op_type else "proper"
            desc = get_operation_description(rot, trans)
            jones = rotation_to_jones(rot, trans)
            lines.append(f"{i:<10} {type_str:<15} {desc:<25} {jones:<20}")
        return "\n".join(lines)


@dataclass
class ChiralTransition:
    """Complete information about a chiral phase transition."""

    parent_spg_number: int
    parent_spg_symbol: str
    parent_spg_order: int

    qpoint: np.ndarray
    qpoint_label: str
    irrep_label: str
    irrep_dimension: int

    opd: OrderParameterDirection

    daughter_spg_number: int
    daughter_spg_symbol: str
    daughter_spg_order: int

    domain_multiplicity: int
    enantiomeric_domain_count: int

    lost_operations: list[LostOperation]
    lost_inversion: bool
    lost_mirrors: int
    lost_glides: int

    sohncke_class: SohnckeClass
    enantiomorph_partner: Optional[int]

    def get_summary(self) -> str:
        """Return one-line summary."""
        return (
            f"{self.parent_spg_symbol}(#{self.parent_spg_number}) "
            f"--[{self.irrep_label} @ {self.qpoint_label}]--> "
            f"{self.daughter_spg_symbol}(#{self.daughter_spg_number}) "
            f"[domains={self.domain_multiplicity}]"
        )


# =============================================================================
# OPD Symbolic Conversion
# =============================================================================

def opd_to_symbolic(opd: np.ndarray, tolerance: float = 1e-10) -> str:
    """
    Convert numerical OPD to symbolic notation.
    Handles complex numbers like (a, ia).
    """
    opd = np.atleast_2d(opd)
    num_params, dim = opd.shape
    result = []

    def format_val(v, param_char):
        if np.isclose(v, 0, atol=tolerance):
            return None
        if np.isclose(v, 1, atol=tolerance):
            return param_char
        if np.isclose(v, -1, atol=tolerance):
            return f"-{param_char}"
        if np.isclose(v, 1j, atol=tolerance):
            return f"i{param_char}"
        if np.isclose(v, -1j, atol=tolerance):
            return f"-i{param_char}"
        
        # General case
        if np.isclose(v.imag, 0, atol=tolerance):
            return f"{v.real:.2f}{param_char}"
        elif np.isclose(v.real, 0, atol=tolerance):
            return f"{v.imag:.2f}i{param_char}"
        else:
            return f"({v.real:.2f}+{v.imag:.2f}j){param_char}"

    for d in range(dim):
        col = opd[:, d]
        terms = []
        for idx in range(num_params):
            term = format_val(col[idx], chr(ord('a') + idx))
            if term:
                terms.append(term)
        
        if not terms:
            result.append('0')
        else:
            res_str = '+'.join(terms).replace('+-', '-')
            result.append(res_str)

    return '(' + ','.join(result) + ')'


# =============================================================================
# Main Class: ChiralTransitionFinder
# =============================================================================

def _get_crystal_system_lattice(spg_number: int) -> np.ndarray:
    """Return a generic lattice consistent with the crystal system."""
    if 1 <= spg_number <= 2: # Triclinic
        # Must be generic enough to have only 1/P-1 symmetry
        return np.array([[1.0, 0.05, 0.1], [0.0, 1.07, 0.15], [0.0, 0.0, 1.13]])
    if 3 <= spg_number <= 15: # Monoclinic
        return np.array([[1.0, 0.0, 0.2], [0.0, 1.1, 0.0], [0.0, 0.0, 1.2]])
    if 16 <= spg_number <= 74: # Orthorhombic
        return np.diag([1.0, 1.1, 1.2])
    if 75 <= spg_number <= 142: # Tetragonal
        return np.diag([1.0, 1.0, 1.5])
    if 143 <= spg_number <= 194: # Trigonal/Hexagonal
        # Standard hexagonal basis: a=b, gamma=120
        return np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3)/2, 0.0], [0.0, 0.0, 1.5]])
    if 195 <= spg_number <= 230: # Cubic
        return np.eye(3)
    return np.eye(3)


class ChiralTransitionFinder:
    """
    Find all possible chiral phase transitions from a parent space group.

    This class uses pure group-theoretical analysis (no atomic structures)
    to enumerate all isotropy subgroups of irreps at special q-points and
    identify which lead to chiral Sohncke space groups.
    """

    def __init__(self, spg_number: int, symprec: float = 1e-5):
        """
        Initialize finder with parent space group.

        Args:
            spg_number: International space group number (1-230)
            symprec: Symmetry tolerance for numerical comparisons
        """
        if not HAS_SPGLIB:
            raise ImportError(
                "The 'spglib' package is required. "
                "Install with: pip install spglib"
            )

        if not 1 <= spg_number <= 230:
            raise ValueError(f"Invalid space group number: {spg_number}")

        self.spg_number = spg_number
        self.symprec = symprec
        self._spacegroup_info: Optional[SpaceGroupInfo] = None
        self._special_qpoints: Optional[list[tuple]] = None

    @property
    def spacegroup_info(self) -> SpaceGroupInfo:
        """Get space group information (lazy loading)."""
        if self._spacegroup_info is None:
            self._spacegroup_info = self._load_spacegroup_info()
        return self._spacegroup_info

    @property
    def is_parent_chiral(self) -> bool:
        """Check if parent space group is already chiral (Sohncke)."""
        return is_sohncke(self.spg_number)

    @property
    def is_parent_centrosymmetric(self) -> bool:
        """Check if parent space group has inversion symmetry."""
        return self.spacegroup_info.has_inversion()

    def get_proper_subgroup_info(self) -> tuple[int, str, int]:
        """
        Get information about the proper subgroup (all proper rotations preserved).
        
        This is a fast method that doesn't require irrep tables.
        It returns the subgroup formed by keeping only proper operations (det > 0).
        
        Returns:
            (daughter_spg_number, daughter_spg_symbol, daughter_order)
            Returns (0, "Unknown", 0) if no valid daughter is found.
        """
        info = self.spacegroup_info
        
        proper_indices = [
            i for i, rot in enumerate(info.rotations)
            if np.linalg.det(rot) > 0
        ]
        
        if not proper_indices:
            return 0, "Unknown", 0
        
        daughter_num, daughter_sym, vol_ratio = self._identify_daughter_spacegroup(
            proper_indices, np.zeros(3)
        )
        return daughter_num, daughter_sym, len(proper_indices)

    def _load_spacegroup_info(self) -> SpaceGroupInfo:
        """Load space group symmetry from spglib database."""
        target_hall = 0
        for hall_number in range(1, 531):
            sg_type = spglib.get_spacegroup_type(hall_number)
            if sg_type is not None and _spglib_attr(sg_type, 'number', 0) == self.spg_number:
                # Prefer hexagonal setting for trigonal systems to match our generic lattice
                if 143 <= self.spg_number <= 167:
                    if _spglib_attr(sg_type, 'choice', '') == 'H':
                        target_hall = hall_number
                        break
                    elif target_hall == 0:
                        target_hall = hall_number
                else:
                    target_hall = hall_number
                    break
        
        if target_hall == 0:
            raise RuntimeError(f"Could not load symmetry for space group {self.spg_number}")

        sym = spglib.get_symmetry_from_database(target_hall)
        if sym is None:
            raise RuntimeError(f"Could not load symmetry dataset for hall number {target_hall}")
            
        sg_type = spglib.get_spacegroup_type(target_hall)
        if sg_type is None:
            raise RuntimeError(f"Could not load spacegroup type for hall number {target_hall}")
        
        # Initial info from database
        info = SpaceGroupInfo(
            number=self.spg_number,
            symbol=_spglib_attr(sg_type, 'international_short', ''),
            rotations=np.array(sym['rotations']),
            translations=np.array(sym['translations']),
            point_group_symbol=_spglib_attr(sg_type, 'pointgroup_international', ''),
            order=len(sym['rotations'])
        )
        
        # Get primitive symmetry and lattice
        lattice = _get_crystal_system_lattice(self.spg_number)
        x, y, z = 0.12345, 0.45678, 0.78901
        all_pos_list = []
        for r, t in zip(info.rotations, info.translations):
            all_pos_list.append((np.dot(r, [x, y, z]) + t) % 1.0)
        all_pos_arr = np.array(all_pos_list)
        numbers = np.ones(len(all_pos_arr), dtype='intc')
        cell = (lattice.tolist(), all_pos_arr.tolist(), numbers.tolist())
        
        prim_cell = spglib.find_primitive(cell, symprec=self.symprec)
        if prim_cell:
            info.primitive_lattice = np.array(prim_cell[0])
            # Get symmetry of the primitive cell, but be CAREFUL not to add extra symmetry
            # (like inversion in P1). 
            # Better: Filter the existing operations to find those that map to primitive cell.
            
            # Find centering translations in conventional basis
            centering_trans = []
            for r, t in zip(info.rotations, info.translations):
                if np.allclose(r, np.eye(3)):
                    centering_trans.append(t % 1.0)
            
            # P_inv transforms conventional to primitive
            # L_prim = P L_conv => P = L_prim @ inv(L_conv)
            P = np.dot(info.primitive_lattice, np.linalg.inv(lattice))
            P_inv = np.linalg.inv(P)
            
            # Transformation for coordinates and rotations: M = (P^-1)^T
            M = P_inv.T
            M_inv = P.T
            
            prim_rots = []
            prim_trans = []
            seen_rots: list[np.ndarray] = []
            
            for r, t in zip(info.rotations, info.translations):
                # r_prim = M r_conv M_inv
                r_prim = np.round(np.dot(M, np.dot(r, M_inv))).astype(int)
                
                # Check if we already have this rotation in primitive basis
                is_new = True
                for sr in seen_rots:
                    if np.allclose(r_prim, sr):
                        is_new = False
                        break
                
                if is_new:
                    seen_rots.append(r_prim)
                    prim_rots.append(r_prim)
                    # t_prim = M t_conv
                    prim_trans.append(np.dot(M, t) % 1.0)
            
            info.primitive_rotations = np.array(prim_rots)
            info.primitive_translations = np.array(prim_trans)
        else:
            info.primitive_rotations = info.rotations
            info.primitive_translations = info.translations
            info.primitive_lattice = lattice
            
        return info

    def get_special_qpoints(self) -> list[tuple[np.ndarray, str]]:
        """
        Get all special q-points from BCS irrep tables.

        Returns:
            List of (qpoint, label) tuples
        """
        if not HAS_IRREP:
            raise ImportError(
                "The 'irrep' package is required. "
                "Install with: pip install irrep"
            )

        if self._special_qpoints is None:
            table = IrrepTable(str(self.spg_number), spinor=False)
            self._special_qpoints = [
                (np.array(irr.k, dtype=float), irr.kpname)
                for irr in table.irreps
            ]
        return self._special_qpoints

    def _get_transformation_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get transformation matrices between primitive and conventional bases.
        
        Returns:
            (P, P_inv) where L_prim = P @ L_conv
        """
        info = self.spacegroup_info
        conv_lattice = _get_crystal_system_lattice(self.spg_number)
        
        # P = L_prim @ inv(L_conv)
        P_raw = np.dot(info.primitive_lattice, np.linalg.inv(conv_lattice))
        
        # Clean P: centering matrices always have entries like 0, 1/2, 1/3, 2/3, 1/6
        # Rounding to nearest 1/12 should be very safe
        P = np.round(P_raw * 12) / 12
        P_inv = np.linalg.inv(P)
        return P, P_inv

    def _get_irreptables_basis(self) -> str:
        """
        Determine if irreptables uses conventional or primitive basis for this SG.
        
        Returns:
            "conventional" or "primitive"
        """
        if not HAS_IRREP:
            return "conventional"
            
        table = IrrepTable(str(self.spg_number), spinor=False)
        if not table.symmetries:
            return "conventional"
            
        irr_rot = table.symmetries[0].R # Identity is usually first, but check for first non-ID too
        if np.allclose(irr_rot, np.eye(3)) and len(table.symmetries) > 1:
            irr_rot = table.symmetries[1].R
            
        info = self.spacegroup_info
        
        # Try to match with conventional rotations
        # For non-primitive cells, info.rotations contains centering
        for rot in info.rotations:
            if np.allclose(irr_rot, rot):
                return "conventional"
        
        # Try to match with primitive rotations
        for rot in info.primitive_rotations:
            if np.allclose(irr_rot, rot):
                return "primitive"
        
        # Fallback to order-based check
        num_irr_syms = len(table.symmetries)
        if num_irr_syms == info.order:
            return "conventional"
        elif num_irr_syms == len(info.primitive_rotations):
            return "primitive"
        else:
            return "conventional"

    def get_irreps_at_qpoint(
        self,
        qpoint: np.ndarray,
        qpoint_label: str
    ) -> list[dict]:
        """
        Get all irrep labels and small representation matrices at a q-point.

        Args:
            qpoint: (3,) fractional coordinates (from get_special_qpoints)
            qpoint_label: BCS label

        Returns:
            List of dicts with irrep information
        """
        info = self.spacegroup_info
        
        # Determine if qpoint is already primitive
        irr_basis = self._get_irreptables_basis()
        
        if irr_basis == "primitive":
            qpoint_prim = qpoint
            # qpoint_conv = P_inv @ qpoint_prim
            _, P_inv = self._get_transformation_matrices()
            qpoint_conv = np.dot(P_inv, qpoint_prim)
        else:
            qpoint_conv = qpoint
            # qpoint_prim = P @ qpoint_conv
            P, _ = self._get_transformation_matrices()
            qpoint_prim = np.dot(P, qpoint_conv)


        if HAS_SPGREP:
            try:
                # Always use PRIMITIVE symmetry for spgrep to be consistent
                # with the lattices used in subgroup identification.
                rots_target = info.primitive_rotations
                trans_target = info.primitive_translations
                qpoint_target = qpoint_prim
                
                irreps, mapping = get_spacegroup_irreps_from_primitive_symmetry(
                    rots_target,
                    trans_target,
                    qpoint_target
                )
                rotations = rots_target
                translations = trans_target

                # Try to label them using irreptables
                labels = [f"irrep_{i}" for i in range(len(irreps))]
                if HAS_IRREP:
                    try:
                        table = IrrepTable(str(self.spg_number), spinor=False)
                        target_q_irreps = [irr for irr in table.irreps if irr.kpname == qpoint_label]
                        if not target_q_irreps:
                            target_q_irreps = [
                                irr for irr in table.irreps 
                                if np.allclose((np.array(irr.k) - qpoint + 0.5) % 1.0, 0.5)
                            ]
                        
                        # Use our mapping to match primitive ops with BCS
                        op_mapping, _ = self._get_irreptables_op_mapping(use_primitive=True)
                        P, _ = self._get_transformation_matrices()
                        
                        for i, irrep_mats in enumerate(irreps):
                            spgrep_chars = [np.trace(m) for m in irrep_mats]
                            
                            
                            n_overlaps = {}
                            for irr_target in target_q_irreps:
                                overlap = 0
                                count = 0
                                for j, prim_idx in enumerate(mapping):
                                    # prim_idx is index into info.primitive_rotations
                                    res = op_mapping.get(prim_idx)
                                    if res is None:
                                        continue
                                    
                                    bcs_idx, c = res
                                    char_bcs = irr_target.characters.get(bcs_idx)
                                    if char_bcs is not None:
                                        # Character must be adjusted by the translation difference phase
                                        # t_prim = t_bcs + c
                                        # exp(-i q . t_prim) = exp(-i q . t_bcs) * exp(-i q . c)
                                        phase = np.exp(-2j * np.pi * np.dot(qpoint_prim, c))
                                        overlap += np.conj(char_bcs * phase) * spgrep_chars[j]
                                        count += 1
                                
                                if count > 0:
                                    n = overlap / count
                                    if np.abs(n - np.round(n.real)) < 0.1 and np.round(n.real) >= 1:
                                        n_overlaps[irr_target.name] = int(np.round(n.real))

                            if n_overlaps:
                                # total_dim check ...
                                label_parts = []
                                for name in sorted(n_overlaps.keys()):
                                    n = n_overlaps[name]
                                    if n == 1:
                                        label_parts.append(name)
                                    else:
                                        label_parts.append(f"{n}{name}")
                                labels[i] = "+".join(label_parts)
                    except Exception:
                        pass
                
                results = []
                for i, irrep in enumerate(irreps):
                    results.append({
                        'label': labels[i],
                        'small_rep': irrep,
                        'little_rotations': rotations[mapping],
                        'little_translations': translations[mapping],
                        'mapping': mapping,
                        'dimension': irrep.shape[1]
                    })
                return results
            except Exception:
                pass

        # Fallback to irreptables (only works for 1D or if we only need characters)
        if HAS_IRREP:
            try:
                table = IrrepTable(str(self.spg_number), spinor=False)
                # Find irreps at this q-point
                target_q_irreps = [irr for irr in table.irreps if irr.kpname == qpoint_label]
                if not target_q_irreps:
                    # Try matching by k-vector
                    target_q_irreps = [
                        irr for irr in table.irreps 
                        if np.allclose((np.array(irr.k) - qpoint + 0.5) % 1.0, 0.5)
                    ]
                
                if not target_q_irreps:
                    return []
                
                # We also need to know which parent operations are in the little group
                lg_indices = []
                for i, (rot, trans) in enumerate(zip(info.rotations, info.translations)):
                    rot_q = np.dot(rot.T, qpoint)
                    if np.allclose((rot_q - qpoint + 0.5) % 1.0, 0.5):
                        lg_indices.append(i)
                
                op_mapping, origin_shift = self._get_irreptables_op_mapping()
                
                results = []
                for irr in target_q_irreps:
                    # For 1D irreps, characters ARE the matrices
                    if irr.dim == 1:
                        small_rep = []
                        for idx in lg_indices:
                            mapped = op_mapping.get(idx)
                            if mapped is not None:
                                irr_idx, translation_diff = mapped
                                char = irr.characters.get(irr_idx, 0)
                                # Apply phase factor for any translation difference
                                # phase = exp(-i * 2pi * q * delta_t)
                                # But actually irreptables operations are symmorphic relative to some origin
                                # It's better to use the exact character if it's symmorphic, but we need
                                # to be careful about non-symmorphic parts.
                                # Assuming delta_t handles this:
                                phase = np.exp(-2j * np.pi * np.dot(qpoint, translation_diff))
                                small_rep.append(np.array([[char * phase]]))
                            else:
                                small_rep.append(np.array([[0]]))
                        
                        results.append({
                            'label': irr.name,
                            'small_rep': np.array(small_rep),
                            'little_rotations': info.rotations[lg_indices],
                            'little_translations': info.translations[lg_indices],
                            'mapping': lg_indices,
                            'dimension': 1
                        })
                    else:
                        # For multi-dimensional irreps, we can't easily get matrices
                        # but we can return the characters for later use?
                        # For now, just mark it as dimension > 1
                        results.append({
                            'label': irr.name,
                            'small_rep': None, # No matrices for dim > 1
                            'little_rotations': info.rotations[lg_indices],
                            'little_translations': info.translations[lg_indices],
                            'mapping': lg_indices,
                            'dimension': irr.dim
                        })
                return results
            except Exception:
                return []
                
        return []

    def get_irrep_labels_at_qpoint(
        self,
        qpoint: np.ndarray,
        qpoint_label: str
    ) -> list[dict]:
        """
        Get irrep labels and character tables at a q-point using irrep package.

        This method uses only the irrep package (no spgrep required).

        Args:
            qpoint: (3,) fractional coordinates
            qpoint_label: BCS label

        Returns:
            List of dicts with 'label', 'dimension', 'characters' keys
        """
        if not HAS_IRREP:
            raise ImportError(
                "The 'irrep' package is required. "
                "Install with: pip install irrep"
            )

        sg = SpaceGroupIrreps.from_cell(
            cell=(np.eye(3), [[0, 0, 0]], [1]),
            spinor=False,
            include_TR=False,
            search_cell=True,
            symprec=self.symprec
        )

        try:
            bcs_table = sg.get_irreps_from_table(qpoint_label, qpoint)
        except Exception:
            return []

        results = []
        for label, char_dict in bcs_table.items():
            char_values = list(char_dict.values())
            dim = int(round(abs(char_values[0]))) if char_values else 1

            results.append({
                'label': label,
                'dimension': dim,
                'characters': char_dict,
            })

        return results

    def _get_irreptables_op_mapping(self, use_primitive: bool = False) -> tuple[dict[int, tuple[int, np.ndarray]], np.ndarray]:
        """
        Create a mapping from spglib symmetry indices to irreptables symmetry indices.
        
        Args:
            use_primitive: If True, map from info.primitive_rotations instead of info.rotations.
            
        Returns:
            (mapping, shift) where:
            - mapping: Dict mapping spglib_idx -> (irreptables_idx (1-based), translation_diff_primitive)
            - shift: Origin shift found (conventional basis)
        """
        if not HAS_IRREP:
            return {}, np.zeros(3)
            
        parent_info = self.spacegroup_info
        table = IrrepTable(str(self.spg_number), spinor=False)
        P, P_inv = self._get_transformation_matrices()
        
        irr_basis = self._get_irreptables_basis()
        
        if use_primitive:
            target_rotations = parent_info.primitive_rotations
            target_translations = parent_info.primitive_translations
            # Target basis is primitive.
            if irr_basis == "primitive":
                possible_MTs = [np.eye(3)]
            else:
                # irreptables is conventional, target is primitive
                # r_conv = P r_prim P_inv
                possible_MTs = [P]
        else:
            target_rotations = parent_info.rotations
            target_translations = parent_info.translations
            # Target basis is conventional.
            if irr_basis == "conventional":
                possible_MTs = [np.eye(3)]
            else:
                # irreptables is primitive, target is conventional
                # r_prim = P_inv r_conv P
                possible_MTs = [P_inv]

        # Origin shifts in target basis
        # Include fractions needed for all crystal systems:
        # - Orthorhombic, tetragonal, cubic: 0, 1/4, 1/2, 3/4
        # - Hexagonal, trigonal: 1/3, 2/3, 1/6
        # - Monoclinic, triclinic: finer sampling (1/8)
        shift_fractions = [0, 1/8, 1/6, 1/4, 1/3, 3/8, 1/2, 5/8, 2/3, 3/4, 5/6, 7/8]
        
        shifts = []
        for i in shift_fractions:
            for j in shift_fractions:
                for k in shift_fractions:
                    shifts.append(np.array([i, j, k]))
        
        for MT in possible_MTs:
            MT_inv = np.linalg.inv(MT)
            for shift in shifts:
                mapping = {}
                for i, (r_target, t_target) in enumerate(zip(target_rotations, target_translations)):
                    expected_t = t_target + np.dot(r_target - np.eye(3), shift)
                    
                    for j, sym_irr in enumerate(table.symmetries, 1):
                        r_expected = np.round(np.dot(MT, np.dot(sym_irr.R, MT_inv))).astype(int)
                        if np.allclose(r_target, r_expected):
                            diff = expected_t - np.dot(MT, sym_irr.t)
                            
                            # Transform diff to primitive basis to check if it's integer
                            if use_primitive:
                                c = diff
                            else:
                                # t_prim = t_conv @ P_inv
                                c = np.dot(diff, P_inv)
                                
                            if np.allclose((c + 0.5) % 1.0, 0.5):
                                mapping[i] = (j, np.round(c).astype(int))
                                break
                    else:
                        break
                
                if len(mapping) == len(target_rotations):
                    return mapping, shift
                    
        # Fallback
        mapping = {}
        for i, r_spg in enumerate(target_rotations):
            for j, sym_irr in enumerate(table.symmetries, 1):
                if np.allclose(r_spg, sym_irr.R):
                    mapping[i] = (j, np.zeros(3, dtype=int))
                    break
        return mapping, np.zeros(3)

    def _get_irrep_label_for_subgroup(
        self, 
        qpoint: np.ndarray, 
        qpoint_label: str, 
        subgroup_indices: list[int]
    ) -> str:
        """Find the irreptables label for an irrep that preserves the subgroup."""
        if not HAS_IRREP:
            return f"{qpoint_label}_proper"
            
        # Determine if qpoint is already primitive
        irr_basis = self._get_irreptables_basis()
        use_prim_mapping = (irr_basis == "primitive")
        op_mapping, shift = self._get_irreptables_op_mapping(use_primitive=use_prim_mapping)
        if not op_mapping:
            return f"{qpoint_label}_proper"
            
        P, _ = self._get_transformation_matrices()
        qpoint_prim = np.dot(P, qpoint)

        table = IrrepTable(str(self.spg_number), spinor=False)
        target_q_irreps = [irr for irr in table.irreps if irr.kpname == qpoint_label]
        
        if not target_q_irreps:
            # Try matching by k-vector
            target_q_irreps = [
                irr for irr in table.irreps 
                if np.allclose((np.array(irr.k) - qpoint + 0.5) % 1.0, 0.5)
            ]
            
        if not target_q_irreps:
            return f"{qpoint_label}_proper"
            
        # We look for a 1D irrep where all subgroup operations that are in little group 
        # have character matching the translation phase.
        
        # Little group indices at q (in conventional basis as provided to this method)
        lg_indices = []
        info = self.spacegroup_info
        target_rots = info.primitive_rotations if use_prim_mapping else info.rotations
        target_trans = info.primitive_translations if use_prim_mapping else info.translations
        
        # Transform qpoint to the basis of target_rots/trans
        if use_prim_mapping:
            q_target = qpoint_prim
        else:
            q_target = qpoint
            
        for i, (rot, trans) in enumerate(zip(target_rots, target_trans)):
            rot_q = np.dot(rot.T, q_target)
            if np.allclose((rot_q - q_target + 0.5) % 1.0, 0.5):
                lg_indices.append(i)
        
        subgroup_set = set(subgroup_indices)
        
        best_label = f"{qpoint_label}_proper"
        for irr in target_q_irreps:
            if irr.dim != 1:
                continue
                
            match = True
            for idx in lg_indices:
                res = op_mapping.get(idx)
                if res is None:
                    continue
                
                irr_idx, c = res
                if irr_idx not in irr.characters:
                    continue
                
                char_target = irr.characters[irr_idx]
                phase = np.exp(-2j * np.pi * np.dot(qpoint_prim, c))
                char_target_with_phase = char_target * phase
                
                # The expected character in parent is exp(-2pi i q.t)
                trans = target_trans[idx]
                expected_char = np.exp(-2j * np.pi * np.dot(q_target, trans))
                
                if idx in subgroup_set:
                    if not np.isclose(char_target_with_phase, expected_char):
                        match = False
                        break
                else:
                    if np.isclose(char_target_with_phase, expected_char):
                        match = False
                        break
            
            if match:
                return irr.name
                
        return best_label

    def enumerate_isotropy_subgroups(
        self,
        qpoint: np.ndarray,
        small_rep: np.ndarray,
        little_rotations: np.ndarray,
        little_translations: np.ndarray
    ) -> list[tuple[list[int], np.ndarray, str]]:
        """
        Enumerate all maximal isotropy subgroups for an irrep.

        Args:
            qpoint: (3,) q-point coordinates
            small_rep: (order, dim, dim) small representation
            little_rotations: (order, 3, 3) little group rotations
            little_translations: (order, 3) little group translations

        Returns:
            List of (subgroup_indices, opd_numerical, opd_symbolic) tuples
        """
        if not HAS_SPGREP_MODULATION:
            raise ImportError(
                "The 'spgrep-modulation' package is required. "
                "Install with: pip install spgrep-modulation"
            )

        enumerator = IsotropyEnumerator(
            little_rotations=little_rotations,
            little_translations=little_translations,
            qpoint=qpoint,
            small_rep=small_rep,
        )

        results = []
        for i, indices in enumerate(enumerator.maximal_isotropy_subgroups):
            opds = enumerator.order_parameter_directions[i]
            for opd_num in opds:
                opd_sym = opd_to_symbolic(opd_num)
                results.append((list(indices), opd_num, opd_sym))

        return results

    def _get_star_of_k(self, qpoint_prim: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
        """
        Calculate the star of a k-point and their coset representatives.

        Args:
            qpoint_prim: (3,) fractional coordinates in primitive reciprocal space

        Returns:
            (star, coset_reps) where star is a list of wavevectors and 
            coset_reps are indices of the primitive rotations mapping qpoint_prim to them.
        """
        star: list[np.ndarray] = []
        coset_reps: list[int] = []
        info = self.spacegroup_info

        for i, r in enumerate(info.primitive_rotations):
            # Reciprocal vectors transform by R^-T
            q_new = np.dot(np.linalg.inv(r).T, qpoint_prim)
            
            is_new = True
            for sq in star:
                if np.allclose((q_new - sq + 0.5) % 1.0, 0.5):
                    is_new = False
                    break
            
            if is_new:
                star.append(q_new)
                coset_reps.append(i)
                
        return star, coset_reps

    def _build_induced_representation(
        self, 
        qpoint_prim: np.ndarray, 
        star: list[np.ndarray], 
        coset_reps: list[int], 
        small_rep: np.ndarray, 
        mapping: np.ndarray
    ) -> list[np.ndarray]:
        """
        Construct the full representation matrices induced from the small representation.

        Args:
            qpoint_prim: (3,) fractional coordinates in primitive reciprocal space
            star: List of wavevectors forming the star of k
            coset_reps: Indices of primitive rotations mapping k1 to the star arms
            small_rep: Matrices of the small representation for the little group
            mapping: Indices of little group operations in the full primitive operations

        Returns:
            List of full representation block matrices for every operation in the parent group.
        """
        info = self.spacegroup_info
        lg_rots = info.primitive_rotations[mapping]
        lg_trans = info.primitive_translations[mapping]
        
        def get_lg_index(r_target, t_target):
            for i, (r, t) in enumerate(zip(lg_rots, lg_trans)):
                if np.allclose(r_target, r):
                    dt = (t_target - t) % 1.0
                    dt = np.where(dt > 0.5, dt - 1.0, dt)
                    if np.allclose(dt, 0):
                        return i
            return -1

        m = len(star)
        d = small_rep.shape[1]
        full_reps = []

        for g_idx in range(len(info.primitive_rotations)):
            r_g = info.primitive_rotations[g_idx]
            t_g = info.primitive_translations[g_idx]
            
            mat = np.zeros((m * d, m * d), dtype=complex)
            
            for i, gi_idx in enumerate(coset_reps):
                r_gi = info.primitive_rotations[gi_idx]
                t_gi = info.primitive_translations[gi_idx]
                
                # g_i^-1
                r_gi_inv = np.linalg.inv(r_gi).astype(int)
                t_gi_inv = -np.dot(r_gi_inv, t_gi)
                
                for j, gj_idx in enumerate(coset_reps):
                    r_gj = info.primitive_rotations[gj_idx]
                    t_gj = info.primitive_translations[gj_idx]
                    
                    # g_i^-1 * g * g_j
                    r_tmp = np.dot(r_g, r_gj)
                    t_tmp = np.dot(r_g, t_gj) + t_g
                    
                    r_final = np.dot(r_gi_inv, r_tmp)
                    t_final = np.dot(r_gi_inv, t_tmp) + t_gi_inv
                    
                    lg_idx = get_lg_index(r_final, t_final)
                    if lg_idx != -1:
                        # Add translation phase factor
                        n = t_final - lg_trans[lg_idx]
                        phase = np.exp(-2j * np.pi * np.dot(qpoint_prim, n))
                        mat[i*d:(i+1)*d, j*d:(j+1)*d] = small_rep[lg_idx] * phase
                        
            full_reps.append(mat)
            
        return full_reps

    def _enumerate_multi_k_isotropy_subgroups(
        self,
        qpoint_prim: np.ndarray,
        small_rep: np.ndarray,
        mapping: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[list[int], np.ndarray, str]]]:
        """
        Enumerate isotropy subgroups for multi-dimensional full induced representations.
        
        Args:
            qpoint_prim: Primary wavevector in primitive basis
            small_rep: Small representation matrices
            mapping: Indices of little group operations
            
        Returns:
            (star, full_reps, list of (subgroup_indices, opd_num, opd_sym))
        """
        import itertools
        
        star, coset_reps = self._get_star_of_k(qpoint_prim)
        
        # If star size is 1, small_rep is already the full_rep.
        if len(star) == 1:
            shape = small_rep[0].shape if len(small_rep) > 0 else (1, 1)
            full_reps = [np.zeros(shape) for _ in range(len(self.spacegroup_info.primitive_rotations))]
            for idx_lg, idx_full in enumerate(mapping):
                full_reps[idx_full] = small_rep[idx_lg]
        else:
            full_reps = self._build_induced_representation(
                qpoint_prim, star, coset_reps, small_rep, mapping
            )
            
        dim = len(star) * small_rep.shape[1]
        
        # We only generate multi-k OPDs up to dimension 6 to avoid combinatoric explosion
        # Note: 2*3 = 6 for R-lattice zone boundary points
        if dim > 6:
            return star, full_reps, []
            
        seen_rays = set()
        opds = []
        for vals in itertools.product([-1, 0, 1], repeat=dim):
            v = np.array(vals, dtype=float)
            if np.allclose(v, 0):
                continue
            
            # Normalize sign for unique ray
            idx = np.where(np.abs(v) > 1e-5)[0][0]
            if v[idx] < 0:
                v = -v
                
            key = tuple(v)
            if key not in seen_rays:
                seen_rays.add(key)
                opds.append(v)
                
        results = []
        # Calculate full isotropy subgroup size for each OPD
        for opd in opds:
            subgroup_indices = []
            for g_idx, mat in enumerate(full_reps):
                if mat is not None:
                    # Check invariant: D(g) v = v
                    v_new = np.dot(mat, opd)
                    if np.linalg.norm(v_new - opd) < 1e-5:
                        subgroup_indices.append(g_idx)
            
            if subgroup_indices:
                results.append((subgroup_indices, opd, opd_to_symbolic(opd)))
                
        return star, full_reps, results

    def _identify_daughter_spacegroup(
        self,
        subgroup_indices: Optional[list[int]],
        qpoint: np.ndarray,
        small_rep: Optional[np.ndarray] = None,
        opd: Optional[np.ndarray] = None,
        little_rots: Optional[np.ndarray] = None,
        little_trans: Optional[np.ndarray] = None,
        full_reps: Optional[list[np.ndarray]] = None,
        star: Optional[list[np.ndarray]] = None
    ) -> tuple[int, str, float]:
        info = self.spacegroup_info
        P, P_inv = self._get_transformation_matrices()
        qpoint_prim = np.dot(P, qpoint)

        denoms, S_prim, S_prim_inv, lattice = self._build_supercell(P, qpoint_prim, star, info)

        M = max(denoms) + 2
        lattice_trans_n = get_lattice_translations_for_supercell(S_prim_inv, M)

        if full_reps is not None and star is not None and opd is not None:
            sc_rots, sc_trans = self._sc_ops_from_full_reps(
                S_prim, S_prim_inv, lattice_trans_n, star, full_reps, opd
            )
        elif little_rots is not None and little_trans is not None:
            sc_rots, sc_trans = self._sc_ops_from_little_group(
                S_prim, S_prim_inv, lattice_trans_n, qpoint_prim,
                little_rots, little_trans, small_rep, opd
            )
        elif subgroup_indices is not None and len(subgroup_indices) > 0:
            return self._identify_sg_from_subgroup_indices(
                S_prim, S_prim_inv, lattice_trans_n, lattice, qpoint_prim,
                subgroup_indices, info
            )
        else:
            return 0, "Unknown", 1.0

        return self._identify_sg_from_sc_ops(sc_rots, sc_trans, lattice, S_prim, info)

    def _build_supercell(self, P, qpoint_prim, star, info):
        """Compute supercell diagonal matrix for the given q-point (or star)."""
        import math
        denoms = [1, 1, 1]
        k_points = star if star is not None else [qpoint_prim]
        for k_pt in k_points:
            k_conv = np.dot(P.T, k_pt)
            for i, x in enumerate(k_conv):
                if np.isclose(x, 0, atol=1e-5):
                    continue
                for d in range(1, 13):
                    if np.isclose((x * d) % 1.0, 0, atol=1e-5):
                        denoms[i] = abs(denoms[i] * d) // math.gcd(denoms[i], d)
                        break
        S_prim = np.diag(denoms)
        S_prim_inv = np.linalg.inv(S_prim)
        lattice = np.dot(S_prim, info.primitive_lattice)
        return denoms, S_prim, S_prim_inv, lattice

    def _sc_ops_from_full_reps(self, S_prim, S_prim_inv, lattice_trans_n, star, full_reps, opd):
        """Build supercell ops using full multi-k representations."""
        info = self.spacegroup_info
        dim_small_rep = full_reps[0].shape[0] // len(star)
        sc_rots, sc_trans = [], []
        for j in range(len(info.primitive_rotations)):
            r = info.primitive_rotations[j]
            t = info.primitive_translations[j]
            mat_j = full_reps[j]
            if mat_j is None:
                continue
            for n in lattice_trans_n:
                phase_mat = np.zeros_like(mat_j, dtype=complex)
                for idx_star, k in enumerate(star):
                    phase = np.exp(-2j * np.pi * np.dot(k, n))
                    s = idx_star * dim_small_rep
                    phase_mat[s:s + dim_small_rep, s:s + dim_small_rep] = np.eye(dim_small_rep) * phase
                mat = np.dot(phase_mat, mat_j)
                if np.linalg.norm(np.dot(mat, opd) - opd) < 1e-5:
                    sc_rots.append(np.dot(S_prim_inv, np.dot(r, S_prim)))
                    sc_trans.append(np.dot(S_prim_inv, t + n))
        return sc_rots, sc_trans

    def _sc_ops_from_little_group(self, S_prim, S_prim_inv, lattice_trans_n, qpoint_prim,
                                   little_rots, little_trans, small_rep, opd):
        """Build supercell ops from the little group (with or without an irrep)."""
        sc_rots, sc_trans = [], []
        if small_rep is not None and opd is not None:
            for j in range(len(little_rots)):
                r = little_rots[j]
                t = little_trans[j]
                mat_j = small_rep[j]
                for n in lattice_trans_n:
                    phase = np.exp(-2j * np.pi * np.dot(qpoint_prim, n))
                    if np.linalg.norm(np.dot(mat_j * phase, opd) - opd) < 1e-5:
                        sc_rots.append(np.dot(S_prim_inv, np.dot(r, S_prim)))
                        sc_trans.append(np.dot(S_prim_inv, t + n))
        else:
            for r, t in zip(little_rots, little_trans):
                for n in lattice_trans_n:
                    sc_rots.append(np.dot(S_prim_inv, np.dot(r, S_prim)))
                    sc_trans.append(np.dot(S_prim_inv, t + n))
        return sc_rots, sc_trans

    def _identify_sg_from_subgroup_indices(self, S_prim, S_prim_inv, lattice_trans_n, lattice,
                                            qpoint_prim, subgroup_indices, info):
        """Identify daughter SG from subgroup indices, with zone-boundary best-shift search."""
        target_rots = info.primitive_rotations[subgroup_indices]
        target_trans = info.primitive_translations[subgroup_indices]

        if np.allclose(qpoint_prim, 0):
            sc_rots = [np.dot(S_prim_inv, np.dot(r, S_prim)) for r in target_rots]
            sc_trans = [np.dot(S_prim_inv, t) for t in target_trans]
            return self._identify_sg_from_sc_ops(sc_rots, sc_trans, lattice, S_prim, info)

        # Zone-boundary: try all lattice shifts and pick the best Sohncke SG
        best_num, best_sym, best_vol_ratio = 0, "Unknown", 1.0
        for n_shift in lattice_trans_n:
            sc_rots = [np.dot(S_prim_inv, np.dot(r, S_prim)) for r in target_rots]
            sc_trans = [np.dot(S_prim_inv, t + n_shift) for t in target_trans]
            try:
                rots_arr = np.round(sc_rots).astype('intc')
                trans_arr = np.array(sc_trans, dtype='double')
                sg_type = spglib.get_spacegroup_type_from_symmetry(
                    rots_arr, trans_arr, lattice=lattice, symprec=self.symprec
                )
                if sg_type is not None:
                    num = _spglib_attr(sg_type, 'number', 0)
                    if is_sohncke(num) and num > best_num:
                        best_num = num
                        best_sym = _spglib_attr(sg_type, 'international_short', '')
                        x, y, z = 0.123, 0.456, 0.789
                        all_pos = np.array([(np.dot(r_op, [x, y, z]) + t_op) % 1.0
                                            for r_op, t_op in zip(rots_arr, trans_arr)])
                        numbers = np.ones(len(all_pos), dtype='intc')
                        prim_cell = spglib.find_primitive((lattice, all_pos, numbers), symprec=self.symprec)
                        if prim_cell:
                            best_vol_ratio = np.abs(np.linalg.det(prim_cell[0])) / np.abs(np.linalg.det(info.primitive_lattice))
            except Exception:
                pass
        return best_num, best_sym, best_vol_ratio

    def _identify_sg_from_sc_ops(self, sc_rots, sc_trans, lattice, S_prim, info):
        """Run spglib on the collected supercell ops and return (num, sym, vol_ratio)."""
        if not sc_rots:
            return 0, "Unknown", 1.0
        sc_rots_arr = np.round(sc_rots).astype('intc')
        sc_trans_arr = np.array(sc_trans, dtype='double')
        try:
            sg_type = spglib.get_spacegroup_type_from_symmetry(
                sc_rots_arr, sc_trans_arr, lattice=lattice, symprec=self.symprec
            )
            if sg_type is not None:
                num = _spglib_attr(sg_type, 'number', 0)
                symb = _spglib_attr(sg_type, 'international_short', '')
                x, y, z = 0.123, 0.456, 0.789
                all_pos = np.array([(np.dot(r_op, [x, y, z]) + t_op) % 1.0
                                    for r_op, t_op in zip(sc_rots_arr, sc_trans_arr)])
                numbers = np.ones(len(all_pos), dtype='intc')
                prim_cell = spglib.find_primitive((lattice, all_pos, numbers), symprec=self.symprec)
                if prim_cell:
                    return num, symb, np.abs(np.linalg.det(prim_cell[0])) / np.abs(np.linalg.det(info.primitive_lattice))
                return num, symb, np.abs(np.linalg.det(S_prim))
        except Exception:
            pass
        return 0, "Unknown", 1.0

    def _analyze_lost_operations(
        self,
        parent_rot: np.ndarray,
        parent_trans: np.ndarray,
        daughter_rot: np.ndarray,
        daughter_trans: np.ndarray
    ) -> list[LostOperation]:
        """Identify improper operations lost in the transition."""
        lost = []

        daughter_ops = set()
        for r, t in zip(daughter_rot, daughter_trans):
            key = (tuple(r.flatten()), tuple(np.round(t, 6)))
            daughter_ops.add(key)

        for r, t in zip(parent_rot, parent_trans):
            op_type = classify_improper_operation(r, t)
            if op_type is None:
                continue

            key = (tuple(r.flatten()), tuple(np.round(t, 6)))
            if key not in daughter_ops:
                lost.append(LostOperation(
                    operation_type=op_type,
                    rotation=r.copy(),
                    translation=t.copy(),
                    description=get_operation_description(r, t),
                    jones_symbol=rotation_to_jones(r, t)
                ))

        return lost

    def _count_enantiomeric_domains(
        self,
        lost_ops: list[LostOperation]
    ) -> int:
        """Count enantiomeric domain pairs from lost improper operations.
        
        When an achiral parent transitions to a chiral daughter, spontaneous
        symmetry breaking creates TWO enantiomeric domains (left-handed and
        right-handed). The number of enantiomeric domains is therefore:
        - 0 if no improper operations were lost (parent was already chiral)
        - 2 if any improper operations were lost (achiral -> chiral transition)
        
        Returns:
            0 or 2 (never 1, since enantiomeric pairs always come in twos)
        """
        for op in lost_ops:
            if op.operation_type in (
                ImproperOperationType.INVERSION,
                ImproperOperationType.MIRROR,
                ImproperOperationType.GLIDE
            ):
                return 2  # Enantiomeric pair created by symmetry breaking
        return 0

    def find_chiral_transitions(
        self,
        qpoint: Optional[np.ndarray] = None,
        qpoint_label: Optional[str] = None,
        irrep_label: Optional[str] = None,
        include_non_chiral: bool = False
    ) -> list[ChiralTransition]:
        """
        Find all chiral daughter phases accessible via displacive transitions.

        Args:
            qpoint: If provided, only search this q-point
            qpoint_label: BCS label for the q-point
            irrep_label: If provided, only search this irrep
            include_non_chiral: If True, also include transitions to non-chiral daughters

        Returns:
            List of ChiralTransition objects
        """
        if self.is_parent_chiral:
            raise ValueError(
                f"Parent space group {self.spg_number} is already chiral (Sohncke). "
                "Use a non-chiral space group as parent."
            )

        transitions: list[ChiralTransition] = []
        parent_info = self.spacegroup_info

        if qpoint is not None:
            # We must still find the star of k to be comprehensive
            qp_in = np.array(qpoint)
            
            # Get primitive qpoint for star calculation
            _, P_inv = self._get_transformation_matrices()
            qp_prim = np.dot(P_inv, qp_in)
            
            star, _ = self._get_star_of_k(qp_prim)
            P, _ = self._get_transformation_matrices()
            
            qpoints_to_search = []
            for k_prim in star:
                # L_prim = P L_conv => q_conv = P^T q_prim
                k_conv = np.dot(P.T, k_prim)
                qpoints_to_search.append((k_conv, qpoint_label or "user"))
        else:
            base_qs = self.get_special_qpoints()
            qpoints_to_search = []
            P, _ = self._get_transformation_matrices()
            for qp, label in base_qs:
                _, P_inv = self._get_transformation_matrices()
                qp_prim = np.dot(P_inv.T, qp)
                star, _ = self._get_star_of_k(qp_prim)
                for k_prim in star:
                    k_conv = np.dot(P.T, k_prim)
                    qpoints_to_search.append((k_conv, label))

        # Group by q-point to avoid redundant calculations
        unique_qpoints = []
        seen_q: list[tuple[np.ndarray, str]] = []
        for qp, label in qpoints_to_search:
            # Shift q to [0, 1) range for consistent comparison
            # We use rounding to handle floating point precision
            qp_norm = np.round(qp % 1.0, 8)
            
            is_new = True
            for sq, sl in seen_q:
                # Two q-points are the same if they differ by an integer vector
                # AND have the same label. 
                if np.allclose(qp_norm, np.round(sq % 1.0, 8)):
                    if label == sl:
                        is_new = False
                        break
            
            if is_new:
                unique_qpoints.append((qp.copy(), label))
                seen_q.append((qp.copy(), label))



        for qp, qp_label in unique_qpoints:
            # Determine correct bases
            irr_basis = self._get_irreptables_basis()
            if irr_basis == "primitive":
                qp_prim = qp
                P, P_inv = self._get_transformation_matrices()
                # Reciprocal vectors transformation: q_conv = P^T q_prim
                qp_conv = np.dot(P.T, qp_prim)
            else:
                qp_conv = qp
                _, P_inv = self._get_transformation_matrices()
                # q_prim = (P^-1)^T q_conv
                qp_prim = np.dot(np.linalg.inv(P).T, qp_conv)
            
            try:
                irreps = self.get_irreps_at_qpoint(qp, qp_label)
            except Exception:
                continue

            for irrep_info in irreps:
                if irrep_label is not None and irrep_info['label'] != irrep_label:
                    continue

                # Try full enumeration first
                isotropy_subgroups = []
                # For 1D irreps, use the simple character-based method which gives the
                # FULL isotropy subgroup, not just maximal subgroups
                if irrep_info['dimension'] == 1:
                    # For 1D irrep, character is the representation itself
                    # Isotropy subgroup is all operations where character == 1
                    chars = irrep_info['small_rep'][:, 0, 0]
                    subgroup_indices = np.where(np.abs(chars - 1.0) < 1e-5)[0].tolist()
                    isotropy_subgroups = [(subgroup_indices, np.array([1.0]), "(a)")]
                else:
                    try:
                        isotropy_subgroups = self.enumerate_isotropy_subgroups(
                            qpoint=qp_prim,
                            small_rep=irrep_info['small_rep'],
                            little_rotations=irrep_info['little_rotations'],
                            little_translations=irrep_info['little_translations']
                        )
                    except Exception:
                        # Fallback for 2D irreps when spgrep-modulation fails
                        if irrep_info['dimension'] == 2:
                            # Common OPDs for 2D irreps: (a, 0) and (a, a)
                            for opd_num, opd_sym in [(np.array([1.0, 0.0]), "(a,0)"), (np.array([1.0, 1.0]), "(a,a)")]:
                                # Identify operations preserving this OPD
                                subgroup_indices = []
                                if irrep_info.get('small_rep') is None: continue
                                for j in range(len(irrep_info['small_rep'])):
                                    mat = irrep_info['small_rep'][j]
                                    if np.linalg.norm(np.dot(mat, opd_num) - opd_num) < 1e-5:
                                        subgroup_indices.append(j)
                                if subgroup_indices:
                                    isotropy_subgroups.append((subgroup_indices, opd_num, opd_sym))

                # Augment with multi-k isotropy subgroups
                multi_k_subgroups: list[tuple[list[int], np.ndarray, str]] = []
                multi_k_full_reps = None
                multi_k_star = None
                if 'mapping' in irrep_info:
                    try:
                        multi_k_star, multi_k_full_reps, multi_k_subgroups = self._enumerate_multi_k_isotropy_subgroups(
                            qpoint_prim=qp_prim,
                            small_rep=irrep_info['small_rep'],
                            mapping=irrep_info['mapping']
                        )
                    except Exception:
                        pass

                all_candidates = []
                for sub_idx, opd_num, opd_sym in isotropy_subgroups:
                    all_candidates.append({
                        'type': 'single_k',
                        'subgroup_indices': sub_idx,
                        'opd_num': opd_num,
                        'opd_sym': opd_sym,
                    })
                for sub_idx, opd_num, opd_sym in multi_k_subgroups:
                    all_candidates.append({
                        'type': 'multi_k',
                        'subgroup_indices': sub_idx,
                        'opd_num': opd_num,
                        'opd_sym': opd_sym,
                    })

                for candidate in all_candidates:
                    subgroup_indices_raw: list[int] = candidate['subgroup_indices']  # type: ignore[assignment]
                    cand_opd_num: np.ndarray = candidate['opd_num']  # type: ignore[assignment]
                    cand_opd_sym: str = candidate['opd_sym']  # type: ignore[assignment]
                    c_type: str = candidate['type']  # type: ignore[assignment]

                    if c_type == 'single_k':
                        daughter_rot = irrep_info['little_rotations'][subgroup_indices_raw]
                        daughter_trans = irrep_info['little_translations'][subgroup_indices_raw]
                        try:
                            daughter_num, daughter_sym, vol_ratio = self._identify_daughter_spacegroup(
                                subgroup_indices=None,
                                qpoint=qp_conv,
                                small_rep=irrep_info['small_rep'],
                                opd=cand_opd_num,
                                little_rots=irrep_info['little_rotations'],
                                little_trans=irrep_info['little_translations']
                            )
                        except Exception:
                                        continue
                    else:
                        daughter_rot = parent_info.primitive_rotations[subgroup_indices_raw]
                        daughter_trans = parent_info.primitive_translations[subgroup_indices_raw]
                        try:
                            daughter_num, daughter_sym, vol_ratio = self._identify_daughter_spacegroup(
                                subgroup_indices=None,
                                qpoint=qp_conv,
                                small_rep=None,
                                opd=cand_opd_num,
                                little_rots=parent_info.primitive_rotations,
                                little_trans=parent_info.primitive_translations,
                                full_reps=multi_k_full_reps,
                                star=multi_k_star
                            )
                        except Exception:
                                        continue

                    if daughter_num == 0:
                        continue

                    if not include_non_chiral and not is_sohncke(daughter_num):
                        continue

                    # Success! Create transition
                    daughter_info = ChiralTransitionFinder(daughter_num).spacegroup_info
                    
                    # Domain multiplicity using primitive cell orders and volume ratio
                    parent_prim_order = len(parent_info.primitive_rotations)
                    daughter_prim_order = len(daughter_info.primitive_rotations)
                    domain_mult = (parent_prim_order / daughter_prim_order) * vol_ratio

                    # Lost operations check against FULL primitive set
                    lost_ops = self._analyze_lost_operations(
                        parent_info.primitive_rotations,
                        parent_info.primitive_translations,
                        daughter_rot,
                        daughter_trans
                    )

                    transition = ChiralTransition(
                        parent_spg_number=self.spg_number,
                        parent_spg_symbol=parent_info.symbol,
                        parent_spg_order=parent_info.order,
                        qpoint=qp_conv.copy(),
                        qpoint_label=qp_label,
                        irrep_label=irrep_info['label'],
                        irrep_dimension=irrep_info['dimension'],
                        opd=OrderParameterDirection(
                            numerical=cand_opd_num.copy() if hasattr(cand_opd_num, 'copy') else np.array(cand_opd_num),
                            symbolic=cand_opd_sym,
                            num_free_params=1 if np.atleast_2d(cand_opd_num).shape[0] == 1 else np.atleast_2d(cand_opd_num).shape[0]
                        ),
                        daughter_spg_number=daughter_num,
                        daughter_spg_symbol=daughter_sym,
                        daughter_spg_order=daughter_info.order,
                        domain_multiplicity=int(round(domain_mult)),
                        enantiomeric_domain_count=self._count_enantiomeric_domains(lost_ops),
                        lost_operations=lost_ops,
                        lost_inversion=any(op.operation_type == ImproperOperationType.INVERSION for op in lost_ops),
                        lost_mirrors=sum(1 for op in lost_ops if op.operation_type == ImproperOperationType.MIRROR),
                        lost_glides=sum(1 for op in lost_ops if op.operation_type == ImproperOperationType.GLIDE),
                        sohncke_class=get_sohncke_class(daughter_num),
                        enantiomorph_partner=get_enantiomorph_partner(daughter_num)
                    )
                    
                    # Deduplicate
                    is_duplicate = False
                    for t in transitions:
                        if (t.daughter_spg_number == transition.daughter_spg_number and 
                            t.irrep_label == transition.irrep_label and
                            np.allclose(t.qpoint, transition.qpoint)):
                            
                            # Check if the opd is collinear or identical
                            # Both numerical vectors should be aligned
                            v1 = t.opd.numerical.flatten()
                            v2 = transition.opd.numerical.flatten()
                            
                            if v1.shape != v2.shape:
                                continue
                                
                            norm1 = np.linalg.norm(v1)
                            norm2 = np.linalg.norm(v2)
                            
                            if norm1 > 1e-5 and norm2 > 1e-5:
                                v1_norm = v1 / norm1
                                v2_norm = v2 / norm2
                                dot_product = np.abs(np.dot(v1_norm, v2_norm))
                                
                                # If vectors are collinear, or it's simply a duplicate by symbolic string
                                if np.isclose(dot_product, 1.0, atol=1e-5) or t.opd.symbolic == transition.opd.symbolic:
                                    is_duplicate = True
                                    break
                            elif t.opd.symbolic == transition.opd.symbolic:
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        transitions.append(transition)

            # The "Proper subgroup" fallback logic previously here has been removed 
            # because it is mathematically unphysical for zone boundary q-points.
            # It ignored the fundamental translational symmetry breaking 
            # (the phase factor exp(-iq.t)) required at these points.
            # All valid phase transitions (including chiral ones) are already captured 
            # perfectly by the full isotropy subgroup enumeration above. 


        return transitions


# =============================================================================
# Reporting Functions
# =============================================================================

def format_transition_table(
    transitions: list[ChiralTransition],
    include_non_chiral: bool = False
) -> str:
    """
    Format transitions as a human-readable table, grouped by class.

    Args:
        transitions: List of ChiralTransition objects
        include_non_chiral: If True, also include non-chiral (Class I) transitions

    Returns:
        Formatted string table
    """
    if not transitions:
        return "No transitions found."

    lines = []
    parent = transitions[0]
    width = 130
    lines.append("=" * width)
    lines.append(f"Phase Transitions from {parent.parent_spg_symbol} (#{parent.parent_spg_number})")
    
    # q-point coordinates
    unique_qs = {}
    for t in transitions:
        if t.qpoint_label not in unique_qs:
            unique_qs[t.qpoint_label] = t.qpoint
    
    q_parts = []
    for label, q in unique_qs.items():
        q_parts.append(f"{label}:({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f})")
    lines.append("q-points: " + " | ".join(q_parts))
    lines.append("=" * width)
    lines.append("")

    def _format_subtable(sub_transitions, title, is_chiral=True):
        if not sub_transitions:
            return []
        sub_lines = []
        sub_lines.append(f"--- {title} ---")
        sub_lines.append(f"{'#':>3} {'q-pt':>6} {'Irrep':>24} {'OPD':>12} {'Daughter':>24} {'Screw':>8} {'Mult':>7} {'Lost Ops':>18}")
        sub_lines.append("-" * width)
        for i, t in enumerate(sub_transitions, 1):
            lost_str = f"i={int(t.lost_inversion)},m={t.lost_mirrors},g={t.lost_glides}"
            daughter_str = f"{t.daughter_spg_symbol} (#{t.daughter_spg_number})"
            screw = get_screw_notation(t.daughter_spg_number)
            sub_lines.append(
                f"{i:>3} {t.qpoint_label:>6} {t.irrep_label:>24} "
                f"{t.opd.symbolic:>12} {daughter_str:>24} "
                f"{screw:>8} {t.domain_multiplicity:>7} {lost_str:>18}"
            )
        sub_lines.append("")
        return sub_lines

    class1 = [t for t in transitions if t.sohncke_class == SohnckeClass.CLASS_I]
    class2 = [t for t in transitions if t.sohncke_class == SohnckeClass.CLASS_II]
    class3 = [t for t in transitions if t.sohncke_class == SohnckeClass.CLASS_III]

    if include_non_chiral:
        lines.extend(_format_subtable(class1, "Class I (Non-chiral / Achiral)", is_chiral=False))
    lines.extend(_format_subtable(class2, "Class II (Enantiomorphous pairs)"))
    lines.extend(_format_subtable(class3, "Class III (Chiral-supporting)"))

    lines.append("Legend: i=inversion, m=mirror, g=glide")
    if include_non_chiral:
        lines.append("Class I = Non-chiral; Class II = Enantiomorphous pair; Class III = Chiral-supporting")
    else:
        lines.append("Class II = Enantiomorphous pair; Class III = Chiral-supporting")

    return "\n".join(lines)


def format_lost_operations_detail(transition: ChiralTransition) -> str:
    """
    Format detailed report of lost operations for a transition.

    Args:
        transition: A ChiralTransition object

    Returns:
        Formatted string report
    """
    lines = []
    lines.append(f"Lost operations for {transition.parent_spg_symbol} -> "
                 f"{transition.daughter_spg_symbol}:")
    lines.append("")

    for op in transition.lost_operations:
        lines.append(f"  - {op.operation_type.value}: {op.description}")
        lines.append(f"    Jones: {op.jones_symbol}")

    return "\n".join(lines)
