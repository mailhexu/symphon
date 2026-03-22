"""
Improper symmetry operation classification and Jones-symbol utilities.
"""

from typing import Optional
import numpy as np

from .sohncke import ImproperOperationType, _is_sohncke_from_operations


# =============================================================================
# Helper Functions - Improper Operations
# =============================================================================

def classify_improper_operation(rotation: np.ndarray, translation: Optional[np.ndarray] = None) -> Optional[ImproperOperationType]:
    """
    Classify the type of improper operation from a rotation matrix.

    Args:
        rotation: (3, 3) integer rotation matrix
        translation: (3,) translation vector (optional, needed to distinguish mirror/glide)

    Returns:
        ImproperOperationType or None if proper operation (det=+1)
    """
    det = int(round(np.linalg.det(rotation)))
    if det > 0:
        return None

    trace = int(round(np.trace(rotation)))

    if trace == -3:
        return ImproperOperationType.INVERSION

    if trace == 1:
        if translation is not None and not np.allclose(translation % 1.0, 0, atol=1e-5):
            return ImproperOperationType.GLIDE
        return ImproperOperationType.MIRROR

    return ImproperOperationType.ROTOUNVERSION


def has_improper_operations(rotations: np.ndarray) -> bool:
    """
    Check if a set of rotations contains any improper operations.

    Args:
        rotations: (order, 3, 3) array of rotation matrices

    Returns:
        True if any operation has det < 0
    """
    return not _is_sohncke_from_operations(rotations)


def get_operation_description(rotation: np.ndarray, translation: np.ndarray) -> str:
    """
    Generate human-readable description of a symmetry operation.

    Args:
        rotation: (3, 3) rotation matrix
        translation: (3,) translation vector

    Returns:
        Description string
    """
    op_type = classify_improper_operation(rotation, translation)
    trace = int(round(np.trace(rotation)))
    det = int(round(np.linalg.det(rotation)))

    if det == -1:
        if op_type == ImproperOperationType.INVERSION:
            if np.allclose(translation % 1.0, 0, atol=1e-5):
                return "1-bar (Inversion)"
            return "1-bar (Inversion with shift)"
        elif op_type in (ImproperOperationType.MIRROR, ImproperOperationType.GLIDE):
            if op_type == ImproperOperationType.MIRROR:
                return "m (Mirror plane)"
            
            # Identify glide type
            t = translation % 1.0
            components = []
            if not np.isclose(t[0], 0, atol=1e-5): components.append("a")
            if not np.isclose(t[1], 0, atol=1e-5): components.append("b")
            if not np.isclose(t[2], 0, atol=1e-5): components.append("c")
            
            if len(components) == 1:
                return f"{components[0]} (Glide plane)"
            elif len(components) >= 2:
                # Check for n-glide (1/2, 1/2, 0) etc.
                if all(np.isclose(t[i], 0.5, atol=1e-2) or np.isclose(t[i], 0, atol=1e-2) for i in range(3)):
                    return "n (Glide plane)"
                return "d (Glide plane)"
            return "glide (Glide plane)"
        else:
            trace_to_label = {-1: "4-bar", 0: "3-bar", -2: "6-bar"}
            label = trace_to_label.get(trace, f"{-trace}-bar")
            return f"{label} (Rotoinversion)"
    else:
        # Proper rotations: trace = 1 + 2*cos(2pi/n)
        trace_to_n = {3: 1, 2: 6, 1: 4, 0: 3, -1: 2}
        n = trace_to_n.get(trace, 0)
        
        if n == 1:
            return "1 (Identity)"
        
        # Check for screw axis
        if np.allclose(translation % 1.0, 0, atol=1e-5):
            return f"{n} (Rotation)"
        
        # Calculate k for n_k screw axis
        # T_total = (sum_{i=0}^{n-1} R^i) * t
        sum_R = np.eye(3, dtype=int)
        R_i = rotation.copy()
        for _ in range(n - 1):
            sum_R = sum_R + R_i
            R_i = np.dot(R_i, rotation)
        
        T_total = np.dot(sum_R, translation)
        # Use the maximum component to find k (for axes along lattice vectors)
        k = int(round(np.max(np.abs(T_total)))) % n
        if k == 0: k = n // 2 # Fallback for some centering cases
        
        return f"{n}_{k} (Screw axis)"


def rotation_to_jones(rotation: np.ndarray, translation: np.ndarray) -> str:
    """
    Convert rotation+translation to Jones symbol.

    Args:
        rotation: (3, 3) rotation matrix
        translation: (3,) translation vector

    Returns:
        Jones symbol string, e.g., "x,y,z" or "-x,-y,z+1/2"
    """
    coords = ['x', 'y', 'z']
    result = []

    for i in range(3):
        row = rotation[i, :]
        terms = []
        for j in range(3):
            if row[j] != 0:
                sign = '-' if row[j] < 0 else '+'
                val = abs(row[j])
                if val == 1:
                    terms.append(f"{sign}{coords[j]}")
                else:
                    terms.append(f"{sign}{val:.3f}{coords[j]}")

        s = "".join(terms)
        if s.startswith("+"):
            s = s[1:]
        
        # Handle translation
        t = translation[i] % 1.0
        if not np.isclose(t, 0, atol=1e-5):
            # Try to find common fractions
            found_fraction = False
            for denom in [2, 3, 4, 6, 8, 12]:
                num = int(round(t * denom))
                if np.isclose(t * denom, num, atol=1e-5):
                    if num == 1:
                        s += f"+1/{denom}"
                    else:
                        from math import gcd
                        common = gcd(num, denom)
                        s += f"+{num//common}/{denom//common}"
                    found_fraction = True
                    break
            if not found_fraction:
                s += f"+{t:.3f}"
        
        if not s:
            s = "0"
        result.append(s.replace("+-", "-"))

    return ", ".join(result)
