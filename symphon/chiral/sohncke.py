"""
Sohncke space group classification and cache utilities.

Contains enums, the Sohncke number set derived from spglib, enantiomorphous-pair
tables, pretty-print symbols, and simple cache helpers used by the rest of the
``symphon.chiral`` sub-package.
"""

from typing import Optional
from enum import Enum
from pathlib import Path
import json
import numpy as np

import spglib

# Flags are kept for compatibility if needed, but are now always True
HAS_SPGLIB = True


# =============================================================================
# Enums
# =============================================================================

class ImproperOperationType(Enum):
    """Types of improper symmetry operations."""
    INVERSION = "inversion"
    MIRROR = "mirror"
    GLIDE = "glide"
    ROTOUNVERSION = "rotoinversion"


class SohnckeClass(Enum):
    """Classification of Sohncke space groups."""
    CLASS_I = "achiral"
    CLASS_II = "enantiomorphous"
    CLASS_III = "chiral_supporting"


# =============================================================================
# Sohncke Space Groups - Algorithmically Derived from spglib
# =============================================================================

def _is_sohncke_from_operations(rotations: np.ndarray) -> bool:
    """
    Determine if space group is Sohncke from its rotation operations.

    A Sohncke group contains ONLY proper operations (det=+1).
    Any improper operation (det=-1) makes it non-Sohncke.

    Args:
        rotations: (order, 3, 3) array of rotation matrices

    Returns:
        True if all operations are proper (Sohncke group)
    """
    for rot in rotations:
        det = np.linalg.det(rot)
        if det < 0:
            return False
    return True


def _get_sohncke_numbers_from_spglib() -> frozenset[int]:
    """
    Derive all 65 Sohncke space group numbers from spglib database.

    This function iterates through all Hall symbols (1-530) and checks
    each space group's operations for improper rotations.

    Returns:
        Frozen set of space group numbers that are Sohncke groups
    """
    if not HAS_SPGLIB:
        raise ImportError(
            "spglib is required. Install with: pip install spglib"
        )

    sohncke_numbers = set()

    for hall_number in range(1, 531):
        sg_type = spglib.get_spacegroup_type(hall_number)
        if sg_type is None:
            continue

        sym = spglib.get_symmetry_from_database(hall_number)
        if sym is None:
            continue

        if _is_sohncke_from_operations(sym['rotations']):
            sohncke_numbers.add(sg_type.get('number', 0) if isinstance(sg_type, dict) else getattr(sg_type, 'number', 0))

    return frozenset(sohncke_numbers)


# Pre-compute Sohncke numbers at module load
_SOHNCKE_NUMBERS = _get_sohncke_numbers_from_spglib()

# Re-export for testing purposes if requested via symphon.chiral_transitions shim
SOHNCKE_NUMBERS_PRIVATE = _SOHNCKE_NUMBERS
GET_SOHNCKE_NUMBERS_FROM_SPGLIB_PRIVATE = _get_sohncke_numbers_from_spglib

_ENANTIOMORPHOUS_PAIRS: dict[int, int] = {
    76: 78, 78: 76,
    91: 95, 95: 91,
    92: 96, 96: 92,
    144: 145, 145: 144,
    151: 153, 153: 151,
    152: 154, 154: 152,
    169: 170, 170: 169,
    171: 172, 172: 171,
    178: 179, 179: 178,
    180: 181, 181: 180,
    212: 213, 213: 212,
}

_SOHNCKE_PRETTY_SYMBOLS: dict[int, str] = {
    1: "P 1", 3: "P 2", 4: "P 2_1", 5: "C 2",
    16: "P 2 2 2", 17: "P 2 2 2_1", 18: "P 2_1 2_1 2", 19: "P 2_1 2_1 2_1",
    20: "C 2 2 2_1", 21: "C 2 2 2", 22: "F 2 2 2", 23: "I 2 2 2", 24: "I 2_1 2_1 2_1",
    75: "P 4", 76: "P 4_1", 77: "P 4_2", 78: "P 4_3", 79: "I 4", 80: "I 4_1",
    89: "P 4 2 2", 90: "P 4 2_1 2", 91: "P 4_1 2 2", 92: "P 4_1 2_1 2",
    93: "P 4_2 2 2", 94: "P 4_2 2_1 2", 95: "P 4_3 2 2", 96: "P 4_3 2_1 2",
    97: "I 4 2 2", 98: "I 4_1 2 2",
    143: "P 3", 144: "P 3_1", 145: "P 3_2", 146: "R 3",
    149: "P 3 1 2", 150: "P 3 2 1", 151: "P 3_1 1 2", 152: "P 3_1 2 1",
    153: "P 3_2 1 2", 154: "P 3_2 2 1", 155: "R 3 2",
    168: "P 6", 169: "P 6_1", 170: "P 6_5", 171: "P 6_2", 172: "P 6_4", 173: "P 6_3",
    177: "P 6 2 2", 178: "P 6_1 2 2", 179: "P 6_5 2 2", 180: "P 6_2 2 2", 181: "P 6_4 2 2", 182: "P 6_3 2 2",
    195: "P 2 3", 196: "F 2 3", 197: "I 2 3", 198: "P 2_1 3", 199: "I 2_1 3",
    207: "P 4 3 2", 208: "P 4_2 3 2", 209: "F 4 3 2", 210: "F 4_1 3 2", 211: "I 4 3 2",
    212: "P 4_3 3 2", 213: "P 4_1 3 2", 214: "I 4_1 3 2"
}

_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"


def _get_cache_path(cache_name: str) -> Path:
    """Get path to a cache file."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{cache_name}.json"


def _save_transitions_cache(transitions_data: list[dict]) -> None:
    """Save transition summary to cache."""
    cache_file = _get_cache_path("chiral_transitions_all")
    with open(cache_file, "w") as f:
        json.dump(transitions_data, f)


def _load_transitions_cache() -> Optional[list[dict]]:
    """Load transition summary from cache if it exists."""
    cache_file = _get_cache_path("chiral_transitions_all")
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return None


# =============================================================================
# Helper Functions - Sohncke Classification
# =============================================================================

def is_sohncke(spg_number: int) -> bool:
    """
    Check if space group is one of the 65 Sohncke groups.

    Args:
        spg_number: International space group number (1-230)

    Returns:
        True if the space group is a Sohncke group
    """
    return spg_number in _SOHNCKE_NUMBERS


def get_sohncke_numbers() -> list[int]:
    """
    Return all 65 Sohncke space group numbers.

    Returns:
        Sorted list of Sohncke space group numbers
    """
    return sorted(_SOHNCKE_NUMBERS)


def get_sohncke_class(spg_number: int) -> SohnckeClass:
    """
    Classify a space group into Sohncke classes.

    - CLASS_I: Non-Sohncke (contains improper operations)
    - CLASS_II: Sohncke with enantiomorphous partner (11 pairs)
    - CLASS_III: Sohncke without enantiomorphous partner (43 groups)

    Args:
        spg_number: International space group number (1-230)

    Returns:
        SohnckeClass enum value
    """
    if spg_number not in _SOHNCKE_NUMBERS:
        return SohnckeClass.CLASS_I
    if spg_number in _ENANTIOMORPHOUS_PAIRS:
        return SohnckeClass.CLASS_II
    return SohnckeClass.CLASS_III


def get_enantiomorph_partner(spg_number: int) -> Optional[int]:
    """
    Get the enantiomorphous partner of a Class II Sohncke group.

    Args:
        spg_number: International space group number (1-230)

    Returns:
        Partner space group number, or None if not Class II
    """
    return _ENANTIOMORPHOUS_PAIRS.get(spg_number)


def get_screw_notation(spg_number: int) -> str:
    """
    Get the screw axis notation for a Sohncke space group.
    
    Args:
        spg_number: International space group number (1-230)
        
    Returns:
        Screw notation string (e.g., "4_1", "6_3", "None")
    """
    # Common screw axes in Sohncke groups
    screws = {
        4: "2_1", 17: "2_1", 18: "2_1", 19: "2_1", 20: "2_1", 24: "2_1",
        76: "4_1", 77: "4_2", 78: "4_3", 80: "4_1",
        91: "4_1", 92: "4_1", 93: "4_2", 94: "4_2", 95: "4_3", 96: "4_3", 98: "4_1",
        144: "3_1", 145: "3_2", 151: "3_1", 152: "3_1", 153: "3_2", 154: "3_2",
        169: "6_1", 170: "6_5", 171: "6_2", 172: "6_4", 173: "6_3",
        178: "6_1", 179: "6_5", 180: "6_2", 181: "6_4", 182: "6_3",
        198: "2_1", 199: "2_1", 208: "4_2", 210: "4_1", 212: "4_3", 213: "4_1", 214: "4_1"
    }
    
    res = screws.get(spg_number)
    if res:
        return res
        
    # Attempt to extract from spglib symbol
    try:
        if HAS_SPGLIB:
            # Find a Hall number for this space group to get its standard symbol
            target_hall = 0
            for hall_number in range(1, 531):
                sg_type = spglib.get_spacegroup_type(hall_number)
                if sg_type and (sg_type.get('number', 0) if isinstance(sg_type, dict) else getattr(sg_type, 'number', 0)) == spg_number:
                    target_hall = hall_number
                    break
            
            if target_hall > 0:
                sg_type = spglib.get_spacegroup_type(target_hall)
                symbol = sg_type.get('international_short', '') if isinstance(sg_type, dict) else getattr(sg_type, 'international_short', '')
                import re
                # Match screw axes like 4_2, 3_1, etc. Must have an underscore.
                match = re.search(r'([2346]_(\d))', symbol)
                if match:
                    return match.group(1).replace('_', '_') # Keep underscore for consistency
    except Exception:
        pass
        
    return "None"
