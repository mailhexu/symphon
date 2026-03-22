import spglib
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from symphon.chiral import is_sohncke, get_enantiomorph_partner, SohnckeClass

@dataclass
class MSGChiralityInfo:
    uni_number: int
    bns_number: str
    msg_type: int
    is_chiral: bool
    family_sg_number: int
    sohncke_class: SohnckeClass
    enantiomorph_partner_bns: Optional[str] = None
    enantiomorph_partner_uni: Optional[int] = None

def get_unique_spatial_ops(rots: np.ndarray, trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract unique spatial operations from a magnetic space group."""
    unique_rots = []
    unique_trans = []
    for r, t in zip(rots, trans):
        t_wrap = t % 1.0
        is_dup = False
        for ur, ut in zip(unique_rots, unique_trans):
            if np.array_equal(r, ur) and np.allclose(t_wrap, ut, atol=1e-5):
                is_dup = True
                break
        if not is_dup:
            unique_rots.append(r)
            unique_trans.append(t_wrap)
    return np.array(unique_rots), np.array(unique_trans)

def find_msg_enantiomorph(rots: np.ndarray, trans: np.ndarray, trs: np.ndarray) -> Any:
    """Find the enantiomorphous partner MSG by applying spatial inversion."""
    trans_inv = (-trans) % 1.0
    dataset = spglib.get_magnetic_spacegroup_type_from_symmetry(rots, trans_inv, trs)
    return dataset

def identify_msg_chirality(identifier: str | int) -> MSGChiralityInfo:
    """
    Identify the chirality properties of a given Magnetic Space Group.
    
    Args:
        identifier: A BNS number (str, e.g. '18.16') or a UNI number (int, 1-1651).
        
    Returns:
        MSGChiralityInfo containing detailed chirality classification.
    """
    uni_number = None
    
    # Resolve identifier to UNI number
    if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
        uni_number = int(identifier)
        if not (1 <= uni_number <= 1651):
            raise ValueError(f"UNI number must be between 1 and 1651, got {uni_number}")
    else:
        # Search by BNS number
        bns = str(identifier).strip()
        for i in range(1, 1652):
            t = spglib.get_magnetic_spacegroup_type(i)
            if t.bns_number == bns:
                uni_number = i
                break
        if uni_number is None:
            raise ValueError(f"Could not find Magnetic Space Group with BNS number: {bns}")

    # Get MSG details
    msg_type_info = spglib.get_magnetic_spacegroup_type(uni_number)
    msg_sym = spglib.get_magnetic_symmetry_from_database(uni_number)
    rots = msg_sym['rotations']
    trans = msg_sym['translations']
    trs = msg_sym['time_reversals']
    
    # 1. Is the MSG physically chiral?
    is_chiral = all(np.linalg.det(r) > 0 for r in rots)
    
    # 2. Find Family Space Group
    ur, ut = get_unique_spatial_ops(rots, trans)
    sg_type = spglib.get_spacegroup_type_from_symmetry(ur, ut)
    if sg_type is None:
        raise RuntimeError(f"Could not identify Family Space Group for UNI {uni_number}")
    family_sg = sg_type.number
    
    # 3. Determine Sohncke Class
    if not is_sohncke(family_sg):
        s_class = SohnckeClass.CLASS_I
    else:
        partner = get_enantiomorph_partner(family_sg)
        if partner is not None:
            s_class = SohnckeClass.CLASS_II
        else:
            s_class = SohnckeClass.CLASS_III
            
    # 4. Find Enantiomorph partner MSG if Class II
    partner_bns = None
    partner_uni = None
    if s_class == SohnckeClass.CLASS_II:
        partner_msg = find_msg_enantiomorph(rots, trans, trs)
        if partner_msg is not None:
            partner_bns = partner_msg.bns_number
            partner_uni = partner_msg.uni_number

    return MSGChiralityInfo(
        uni_number=uni_number,
        bns_number=msg_type_info.bns_number,
        msg_type=msg_type_info.type,
        is_chiral=is_chiral,
        family_sg_number=family_sg,
        sohncke_class=s_class,
        enantiomorph_partner_bns=partner_bns,
        enantiomorph_partner_uni=partner_uni
    )
