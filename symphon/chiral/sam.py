import numpy as np
from typing import Optional, Union

class SAMCalculator:
    """
    Calculates Phonon Spin Angular Momentum (SAM).
    
    The SAM for a mode (q, j) is defined as (Zhang & Niu, PRL 112, 085503 (2014)):
    S = hbar * sum_kappa Im(epsilon*_kappa x epsilon_kappa)
    where epsilon_kappa are mass-weighted eigenvectors (phonopy convention).
    """
    @staticmethod
    def calculate(
        displacement: np.ndarray, 
        masses: Optional[np.ndarray] = None,
        mass_weighted: bool = True,
        normalize: bool = False,
        hbar: float = 1.0
    ) -> np.ndarray:
        """
        Calculate SAM 3-vector for a given displacement.
        
        Parameters
        ----------
        displacement : np.ndarray
            Displacement vectors (eigenvectors) of shape (N, 3) or (3N,).
        masses : np.ndarray, optional
            Atomic masses. Required if mass_weighted=False or if calculating 
            geometric circularity.
        mass_weighted : bool
            If True (default), assumes displacement is the mass-weighted 
            eigenvector epsilon (phonopy convention).
            Physical SAM S = hbar * sum_kappa Im(epsilon*_kappa x epsilon_kappa).
            If False, assumes displacement is physical displacement u.
            Physical SAM S = hbar * sum_kappa m_kappa Im(u*_kappa x u_kappa).
        normalize : bool
            If True, normalizes the result by the total kinetic energy or 
            equivalent, ensuring the SAM magnitude is in range [0, hbar].
            If False (default), returns absolute SAM.
        hbar : float
            Reduced Planck constant (default 1.0).
            
        Returns
        -------
        sam : np.ndarray
            SAM 3-vector.
        """
        d = np.array(displacement).reshape(-1, 3)
        
        if mass_weighted:
            # S = hbar * sum_kappa Im(e*_kappa x e_kappa)
            cross = np.cross(d.conj(), d)
            sam = hbar * np.sum(np.imag(cross), axis=0)
            
            if normalize:
                # Normalization factor is sum |e|^2 = 1 (already true for eigenvectors)
                pass
        else:
            if masses is None:
                # Fallback to unweighted if masses not provided
                cross = np.cross(d.conj(), d)
                sam = hbar * np.sum(np.imag(cross), axis=0)
            else:
                # S = hbar * sum_kappa m_kappa Im(u*_kappa x u_kappa)
                cross = np.cross(d.conj(), d)
                weighted_cross = masses[:, np.newaxis] * np.imag(cross)
                sam = hbar * np.sum(weighted_cross, axis=0)
                
                if normalize:
                    # Normalize by sum m|u|^2
                    norm_factor = np.sum(masses * np.sum(np.abs(d)**2, axis=1))
                    if norm_factor > 1e-12:
                        sam = sam / norm_factor

        return sam

    @staticmethod
    def calculate_geometric_circularity(
        displacement: np.ndarray,
        masses: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the 'geometric' circularity, unweighted by mass.
        S_geom = sum_kappa Im(u*_kappa x u_kappa) where sum |u|^2 = 1.
        This matches the formula: sum (1/m_kappa) Im(e*_kappa x e_kappa).
        """
        d = np.array(displacement).reshape(-1, 3)
        # Convert e to u: u = e / sqrt(m)
        u = d / np.sqrt(masses)[:, np.newaxis]
        
        # Normalize u to 1
        norm = np.linalg.norm(u)
        if norm > 1e-12:
            u = u / norm
            
        cross = np.cross(u.conj(), u)
        return np.sum(np.imag(cross), axis=0)

    @staticmethod
    def get_circularity(sam: np.ndarray, axis: Optional[np.ndarray] = None) -> float:
        """
        Extract circularity (scalar) from SAM along a given axis.
        By default, uses the major axis of SAM.
        """
        if axis is not None:
            axis = axis / np.linalg.norm(axis)
            return float(np.dot(sam, axis))
        
        return float(np.linalg.norm(sam))
