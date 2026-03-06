
import json
import numpy as np
from pathlib import Path
from symphon.chiral_transitions import ChiralTransitionFinder, get_sohncke_numbers

def serialize_transition(t):
    # Convert OPD numerical values to real if they are nearly real
    opd_num = t.opd.numerical
    if np.iscomplexobj(opd_num):
        if np.allclose(opd_num.imag, 0, atol=1e-8):
            opd_num = opd_num.real
    
    # Custom recursive conversion for complex numbers in lists/arrays
    def json_ready(obj):
        if isinstance(obj, np.ndarray):
            return [json_ready(x) for x in obj.tolist()]
        if isinstance(obj, list):
            return [json_ready(x) for x in obj]
        if isinstance(obj, complex):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if hasattr(obj, 'item'): # Handle numpy scalars
            item = obj.item()
            if isinstance(item, complex):
                return {"real": float(item.real), "imag": float(item.imag)}
            return item
        return obj

    return {
        "parent_spg_number": int(t.parent_spg_number),
        "parent_spg_symbol": str(t.parent_spg_symbol),
        "qpoint": [float(x) for x in t.qpoint],
        "qpoint_label": str(t.qpoint_label),
        "irrep_label": str(t.irrep_label),
        "irrep_dimension": int(t.irrep_dimension),
        "opd": {
            "numerical": json_ready(opd_num),
            "symbolic": str(t.opd.symbolic)
        },
        "daughter_spg_number": int(t.daughter_spg_number),
        "daughter_spg_symbol": str(t.daughter_spg_symbol),
        "domain_multiplicity": int(t.domain_multiplicity),
        "sohncke_class": t.sohncke_class.value,
        "lost_inversion": bool(t.lost_inversion),
        "lost_mirrors": int(t.lost_mirrors),
        "lost_glides": int(t.lost_glides),
        "lost_ops_indices": sorted([int(op.parent_index) for op in t.lost_operations if op.parent_index is not None]),
        "added_ops_jones": [str(op.jones_symbol) for op in t.added_operations]
    }

def generate_all_transitions(start_sg=1, specific_sgs=None):
    sohncke_set = set(get_sohncke_numbers())
    output_path = Path("symphon/tests/reference_transitions.json")
    
    if output_path.exists():
        with open(output_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    sgs_to_process = specific_sgs if specific_sgs else range(start_sg, 231)
    
    for spg_num in sgs_to_process:
        if specific_sgs is None and str(spg_num) in all_results:
            continue
        if spg_num in sohncke_set:
            continue
            
        print(f"Processing SG {spg_num:3d}...", end=" ", flush=True)
        try:
            finder = ChiralTransitionFinder(spg_num)
            transitions = finder.find_chiral_transitions()
            if transitions:
                all_results[str(spg_num)] = [serialize_transition(t) for t in transitions]
                print(f"Found {len(transitions)}")
            else:
                all_results[str(spg_num)] = []
                print("No chiral transitions")
            
            # Save progress
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error: {e}")
            
    return all_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate reference JSON for chiral transitions.")
    parser.add_argument("--sg", type=int, nargs="+", help="Specific space group numbers to regenerate")
    parser.add_argument("--start", type=int, default=1, help="Start space group number (default: 1)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing entries")
    
    args = parser.parse_args()
    
    output_path = Path("symphon/tests/reference_transitions.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.force and not args.sg:
        if output_path.exists():
            output_path.unlink()
    
    print("Generating chiral transitions for reference...")
    data = generate_all_transitions(start_sg=args.start, specific_sgs=args.sg)
    
    print(f"\nDone. Reference saved to {output_path}")
