import argparse
import sys

from symphon.msg_chiral import identify_msg_chirality
from symphon.chiral_transitions import SohnckeClass

def main():
    parser = argparse.ArgumentParser(
        description="Identify the chirality classification of a Magnetic Space Group (MSG)."
    )
    parser.add_argument(
        "identifier",
        type=str,
        help="BNS number (e.g. '18.16') or UNI number (e.g. 114) of the Magnetic Space Group."
    )
    
    args = parser.parse_args()
    
    try:
        info = identify_msg_chirality(args.identifier)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"==================================================")
    print(f" MAGNETIC SPACE GROUP CHIRALITY ANALYSIS")
    print(f"==================================================")
    print(f"BNS Number        : {info.bns_number}")
    print(f"UNI Number        : {info.uni_number}")
    print(f"Family Space Group: {info.family_sg_number}")
    
    # Magnetic Type
    m_type_str = ""
    if info.msg_type == 1:
        m_type_str = "Type I (Colorless / Purely Spatial)"
    elif info.msg_type == 2:
        m_type_str = "Type II (Grey / Paramagnetic / Diamagnetic)"
    elif info.msg_type == 3:
        m_type_str = "Type III (Black & White, Time-Reversal on Point Group)"
    elif info.msg_type == 4:
        m_type_str = "Type IV (Black & White, Time-Reversal on Translation)"
    print(f"Magnetic Type     : {m_type_str}")
    print(f"--------------------------------------------------")
    
    # Chirality flags
    print(f"Is Chiral?        : {'YES' if info.is_chiral else 'NO'}")
    
    # Sohncke class details
    if info.sohncke_class == SohnckeClass.CLASS_I:
        print(f"Sohncke Class     : Class I (Achiral Family Group)")
    elif info.sohncke_class == SohnckeClass.CLASS_III:
        print(f"Sohncke Class     : Class III (Chiral-supporting, No Enantiomorph Partner)")
    elif info.sohncke_class == SohnckeClass.CLASS_II:
        print(f"Sohncke Class     : Class II (Enantiomorphous pair)")
        print(f"Enantiomorph MSG  : BNS {info.enantiomorph_partner_bns} (UNI {info.enantiomorph_partner_uni})")

if __name__ == "__main__":
    main()
