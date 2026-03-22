import numpy as np
from ..io.phbst import read_phbst_freqs_and_eigvecs, ase_atoms_to_phonopy_atoms
from .core import IrRepsAnaddb
from .phonopy import IrRepsPhonopy, phonopy_load

def get_special_qpoints(primitive_atoms, symprec=1e-5) -> list[dict]:
    """
    Get all special q-points for a given primitive structure.
    Uses the 'irrep' package to find the BCS special q-points and transforms
    them to the input structure's reciprocal basis.
    
    Returns:
        List of dictionaries with keys:
        - label: The BCS q-point label (e.g., 'GM', 'X')
        - qpoint_bcs: The q-point coordinates in the standard BCS reciprocal cell
        - qpoint_input: The q-point coordinates in the input cell's reciprocal basis
    """
    try:
        from irreptables.irreps import IrrepTable
    except ImportError:
        try:
            from irreptables import IrrepTable  # type: ignore
        except ImportError:
            IrrepTable = None

    from irrep.spacegroup_irreps import SpaceGroupIrreps
    
    cell = (primitive_atoms.cell, primitive_atoms.scaled_positions, primitive_atoms.numbers)
    sg = SpaceGroupIrreps.from_cell(
        cell=cell,
        spinor=False,
        include_TR=False,
        search_cell=True,
        symprec=symprec,
        verbosity=0
    )
    
    if IrrepTable is None:
        # Fallback if irreptables is not available
        return []
        
    table = IrrepTable(sg.number_str, spinor=False)
    refUCTinv = np.linalg.inv(sg.refUC.T)
    
    seen_labels = set()
    results = []
    
    for irr in table.irreps:
        if irr.kpname not in seen_labels:
            seen_labels.add(irr.kpname)
            k_bcs = np.array(irr.k, dtype=float)
            k_input = refUCTinv @ k_bcs
            
            # Clean up near-zero values
            k_input = np.where(np.abs(k_input) < 1e-5, 0.0, k_input)
            k_bcs = np.where(np.abs(k_bcs) < 1e-5, 0.0, k_bcs)
            
            results.append({
                "label": irr.kpname,
                "qpoint_bcs": k_bcs.tolist(),
                "qpoint_input": k_input.tolist()
            })
            
    return results


def get_all_irreps_phonopy(
    phonopy_params,
    symprec: float | None = None,
    degeneracy_tolerance: float = 1e-4,
    log_level: int = 0,
) -> dict[str, IrRepsPhonopy]:
    """
    Compute irreps for all special q-points using direct phonopy calculations.
    Returns a dictionary mapping the q-point label to its IrRepsPhonopy instance.
    """
    phonon = phonopy_load(phonopy_params)
    primitive = phonon.primitive
    if symprec is None:
        symprec = phonon._symprec
        
    special_qs = get_special_qpoints(primitive, symprec=symprec)
    results = {}
    
    for sq in special_qs:
        label = sq["label"]
        q_input = sq["qpoint_input"]
        
        irr = IrRepsPhonopy(
            phonopy_params=phonopy_params,
            qpoint=q_input,
            symprec=symprec,
            degeneracy_tolerance=degeneracy_tolerance,
            log_level=log_level
        )
        irr.run(kpname=label)
        results[label] = irr
        
    return results

def find_highsym_qpoints_in_phbst(
    phbst_fname: str,
    symprec: float = 1e-5,
) -> list[dict]:
    """Find high-symmetry q-points that are present in a PHBST file.

    Reads the q-point list from the PHBST file, computes the theoretical
    high-symmetry points for the structure, and returns only those that
    match a q-point in the file.

    Returns:
        List of dicts with keys:
        - ind_q: index of the q-point in the PHBST file (0-based)
        - label: BCS label (e.g. 'GM', 'X')
        - qpoint: fractional coordinates in the input cell basis
    """
    atoms, qpoints, _freqs, _eig_vecs = read_phbst_freqs_and_eigvecs(phbst_fname)
    primitive = ase_atoms_to_phonopy_atoms(atoms)

    try:
        special_qs = get_special_qpoints(primitive, symprec=symprec)
    except Exception:
        special_qs = []

    matched = []
    for sq in special_qs:
        q_ref = np.array(sq["qpoint_input"])
        for ind_q, q in enumerate(qpoints):
            diff = np.array(q) - q_ref
            diff -= np.rint(diff)
            if np.all(np.abs(diff) < symprec):
                matched.append({
                    "ind_q": ind_q,
                    "label": sq["label"],
                    "qpoint": q.tolist(),
                })
                break  # each special point matched at most once

    return matched


def get_all_irreps_anaddb(
    phbst_fname: str,
    symprec: float = 1e-5,
    degeneracy_tolerance: float = 1e-4,
    log_level: int = 0,
) -> dict[int, IrRepsAnaddb]:
    """
    Compute irreps for all q-points present in the anaddb PHBST file.
    Returns a dictionary mapping the q-point index to its IrRepsAnaddb instance.
    """
    atoms, qpoints, freqs, eig_vecs = read_phbst_freqs_and_eigvecs(phbst_fname)
    primitive = ase_atoms_to_phonopy_atoms(atoms)
    
    try:
        special_qs = get_special_qpoints(primitive, symprec=symprec)
    except Exception:
        special_qs = []
        
    results = {}
    for ind_q, q in enumerate(qpoints):
        # Try to match q with a special q-point
        matched_label = None
        for sq in special_qs:
            q_input = np.array(sq["qpoint_input"])
            diff = q - q_input
            diff -= np.rint(diff)
            if np.all(np.abs(diff) < symprec):
                matched_label = sq["label"]
                break
                
        irr = IrRepsAnaddb(
            phbst_fname=phbst_fname,
            ind_q=ind_q,
            symprec=symprec,
            degeneracy_tolerance=degeneracy_tolerance,
            log_level=log_level
        )
        irr.run(kpname=matched_label)
        results[ind_q] = irr
        
    return results
