"""
Generate chiral transition tables for all non-Sohncke space groups.

Usage:
    source .venv/bin/activate && python scripts/generate_all_chiral_tables.py
"""

import sys
from pathlib import Path

# Output directory (absolute path)
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_DIR / "chiral_transition_tables"

from symphon.chiral import (
    ChiralTransitionFinder,
    is_sohncke,
    format_transition_table,
    get_sohncke_numbers,
)

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    sohncke_numbers = get_sohncke_numbers()
    print(f"Sohncke (chiral) space groups: {len(sohncke_numbers)}")
    print(f"Non-Sohncke space groups to process: {230 - len(sohncke_numbers)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    results_summary = []
    
    for spg_num in range(1, 231):
        if is_sohncke(spg_num):
            continue
        
        print(f"Processing SG {spg_num:3d}...", end=" ", flush=True)
        
        try:
            finder = ChiralTransitionFinder(spg_num)
            transitions = finder.find_chiral_transitions()
            
            if transitions:
                output_file = OUTPUT_DIR / f"SG{spg_num:03d}_chiral_transitions.txt"
                table = format_transition_table(transitions)
                
                with open(output_file, 'w') as f:
                    f.write(table)
                    f.write(f"\n\n# Summary\n")
                    f.write(f"# Parent: {transitions[0].parent_spg_symbol} (#{spg_num})\n")
                    f.write(f"# Total chiral transitions: {len(transitions)}\n")
                    f.write(f"# Unique chiral daughters: {len(set(t.daughter_spg_number for t in transitions))}\n")
                
                results_summary.append((spg_num, len(transitions), output_file.name))
                print(f"{len(transitions):3d} transitions -> {output_file.name}")
            else:
                results_summary.append((spg_num, 0, None))
                print("No chiral transitions")
                
        except Exception as e:
            results_summary.append((spg_num, -1, str(e)))
            print(f"ERROR - {e}")
    
    # Write summary
    summary_file = OUTPUT_DIR / "SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write("Chiral Transitions Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Space Groups with Chiral Transitions:\n")
        f.write("-" * 50 + "\n")
        for spg_num, count, filename in results_summary:
            if count > 0:
                f.write(f"  SG {spg_num:3d}: {count:4d} transitions -> {filename}\n")
        
        f.write("\n\nSpace Groups with No Chiral Transitions:\n")
        f.write("-" * 50 + "\n")
        no_trans = [(n, c, fn) for n, c, fn in results_summary if c == 0]
        for spg_num, _, _ in no_trans:
            f.write(f"  SG {spg_num:3d}\n")
        
        f.write("\n\nSpace Groups with Errors:\n")
        f.write("-" * 50 + "\n")
        errors = [(n, c, fn) for n, c, fn in results_summary if c < 0]
        for spg_num, _, error in errors:
            f.write(f"  SG {spg_num:3d}: {error}\n")
        
        f.write("\n\nStatistics:\n")
        f.write("-" * 50 + "\n")
        total_with = sum(1 for _, c, _ in results_summary if c > 0)
        total_without = sum(1 for _, c, _ in results_summary if c == 0)
        total_errors = sum(1 for _, c, _ in results_summary if c < 0)
        total_transitions = sum(c for _, c, _ in results_summary if c > 0)
        
        f.write(f"  Total non-Sohncke groups processed: {len(results_summary)}\n")
        f.write(f"  Groups with chiral transitions: {total_with}\n")
        f.write(f"  Groups without chiral transitions: {total_without}\n")
        f.write(f"  Groups with errors: {total_errors}\n")
        f.write(f"  Total chiral transitions found: {total_transitions}\n")
    
    print(f"\nSummary written to {summary_file}")
    print(f"Total files generated: {sum(1 for _, c, _ in results_summary if c > 0)}")


if __name__ == "__main__":
    main()
