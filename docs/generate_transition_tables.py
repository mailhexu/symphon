#!/usr/bin/env python
"""
Generate chiral transition tables for all non-Sohncke space groups.

Run:
    cd anaddb_irreps/docs
    uv run python generate_transition_tables.py
"""

from symphon.chiral import (
    ChiralTransitionFinder,
    SohnckeClass,
    is_sohncke,
    get_sohncke_numbers,
)
from pathlib import Path
import spglib

OUTPUT_DIR = Path(__file__).parent / "chiral_transitions"


def get_spg_symbol(number: int) -> str:
    """Get space group symbol for a number."""
    # Use ChiralTransitionFinder to get the symbol (it handles Hall number mapping correctly)
    from symphon.chiral import ChiralTransitionFinder
    try:
        finder = ChiralTransitionFinder(number)
        symbol = finder.spacegroup_info.symbol
        return symbol.replace("/", "_").replace(" ", "")
    except Exception:
        return f"SG{number}"


def format_transition_table_md(transitions, parent_num: int, parent_symbol: str) -> str:
    """Format transitions as a markdown table."""
    lines = []
    lines.append(f"# Chiral Transitions from {parent_symbol} (#{parent_num})\n")
    
    if not transitions:
        lines.append("No chiral transitions found for this space group.\n")
        return "\n".join(lines)
    
    # Group by q-point
    qpoints = {}
    for t in transitions:
        key = (t.qpoint_label, tuple(t.qpoint.round(3)))
        if key not in qpoints:
            qpoints[key] = []
        qpoints[key].append(t)
    
    # q-point coordinates
    lines.append("## Q-points\n")
    for (label, qp), _ in qpoints.items():
        lines.append(f"- **{label}**: ({qp[0]:.3f}, {qp[1]:.3f}, {qp[2]:.3f})")
    lines.append("")
    
    # Class II
    class2 = [t for t in transitions if t.sohncke_class == SohnckeClass.CLASS_II]
    if class2:
        lines.append("## Class II (Enantiomorphous Pairs)\n")
        lines.append("These transitions lead to space groups with distinct enantiomorphous partners.\n")
        lines.append("| # | Q-point | Irrep | OPD | Daughter SG | Domains | Lost Ops |")
        lines.append("|---|---------|-------|-----|-------------|---------|----------|")
        for i, t in enumerate(class2, 1):
            lost = f"i={int(t.lost_inversion)},m={t.lost_mirrors},g={t.lost_glides}"
            lines.append(f"| {i} | {t.qpoint_label} | {t.irrep_label} | {t.opd.symbolic} | {t.daughter_spg_symbol} (#{t.daughter_spg_number}) | {t.domain_multiplicity} | {lost} |")
        lines.append("")
    
    # Class III
    class3 = [t for t in transitions if t.sohncke_class == SohnckeClass.CLASS_III]
    if class3:
        lines.append("## Class III (Chiral-Supporting)\n")
        lines.append("These transitions lead to chiral space groups without distinct enantiomorphous partners.\n")
        lines.append("| # | Q-point | Irrep | OPD | Daughter SG | Domains | Lost Ops |")
        lines.append("|---|---------|-------|-----|-------------|---------|----------|")
        for i, t in enumerate(class3, 1):
            lost = f"i={int(t.lost_inversion)},m={t.lost_mirrors},g={t.lost_glides}"
            lines.append(f"| {i} | {t.qpoint_label} | {t.irrep_label} | {t.opd.symbolic} | {t.daughter_spg_symbol} (#{t.daughter_spg_number}) | {t.domain_multiplicity} | {lost} |")
        lines.append("")
    
    lines.append("## Legend\n")
    lines.append("- **i**: inversion lost")
    lines.append("- **m**: mirrors lost")
    lines.append("- **g**: glides lost")
    lines.append("- **Domains**: domain multiplicity")
    
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all space group info
    sohncke_numbers = get_sohncke_numbers()
    print("Generating chiral transition tables...")
    print(f"Sohncke groups (excluded): {len(sohncke_numbers)}")
    
    generated = 0
    skipped_sohncke = 0
    no_transitions = 0
    
    for num in range(1, 231):
        symbol = get_spg_symbol(num)
        filename = f"{num:03d}_{symbol}.md"
        filepath = OUTPUT_DIR / filename
        
        if is_sohncke(num):
            skipped_sohncke += 1
            continue
        
        try:
            finder = ChiralTransitionFinder(num)
            transitions = finder.find_chiral_transitions()
            
            content = format_transition_table_md(transitions, num, symbol)
            filepath.write_text(content)
            
            if transitions:
                generated += 1
                print(f"  {num:3d} {symbol}: {len(transitions)} transitions")
            else:
                no_transitions += 1
                print(f"  {num:3d} {symbol}: no transitions")
                
        except Exception as e:
            print(f"  {num:3d} {symbol}: ERROR - {e}")
    
    print()
    print(f"Summary:")
    print(f"  Generated with transitions: {generated}")
    print(f"  No transitions found: {no_transitions}")
    print(f"  Skipped (Sohncke): {skipped_sohncke}")
    print(f"  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
