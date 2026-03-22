import pytest
from symphon.magnetic import identify_msg_chirality
from symphon.chiral import SohnckeClass

def test_identify_msg_chirality_type_1_class_3():
    # 18.16 is Type 1, Family 18 (Class III)
    info = identify_msg_chirality("18.16")
    assert info.bns_number == "18.16"
    assert info.uni_number == 114
    assert info.family_sg_number == 18
    assert info.msg_type == 1
    assert info.is_chiral is True
    assert info.sohncke_class == SohnckeClass.CLASS_III
    assert info.enantiomorph_partner_bns is None

def test_identify_msg_chirality_type_2_class_2():
    # 76.8 is Type 2, Family 76 (Class II)
    info = identify_msg_chirality("76.8")
    assert info.bns_number == "76.8"
    assert info.uni_number == 668
    assert info.family_sg_number == 76
    assert info.msg_type == 2
    assert info.is_chiral is True
    assert info.sohncke_class == SohnckeClass.CLASS_II
    assert info.enantiomorph_partner_bns == "78.20"
    assert info.enantiomorph_partner_uni == 680

def test_identify_msg_chirality_type_4_class_2():
    # 145.9 is Type 4, Family 144!
    info = identify_msg_chirality("145.9")
    assert info.bns_number == "145.9"
    assert info.uni_number == 1239
    assert info.family_sg_number == 144
    assert info.msg_type == 4
    assert info.is_chiral is True
    assert info.sohncke_class == SohnckeClass.CLASS_II
    assert info.enantiomorph_partner_bns == "144.6"
    assert info.enantiomorph_partner_uni == 1236

def test_identify_msg_chirality_achiral():
    # 2.4 is Type 1, Family 2 (P-1, Achiral)
    info = identify_msg_chirality("2.4")
    assert info.bns_number == "2.4"
    assert info.uni_number == 4
    assert info.family_sg_number == 2
    assert info.msg_type == 1
    assert info.is_chiral is False
    assert info.sohncke_class == SohnckeClass.CLASS_I

def test_invalid_identifier():
    with pytest.raises(ValueError, match="Could not find"):
        identify_msg_chirality("999.999")
        
    with pytest.raises(ValueError, match="must be between 1 and 1651"):
        identify_msg_chirality(2000)
