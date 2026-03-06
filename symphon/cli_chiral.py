#!/usr/bin/env python
"""
CLI tool for finding chiral phase transitions.

Usage:
    anaddb-chiral [options]

Examples:
    # Print transitions for a specific space group
    anaddb-chiral --sg 136
    
    # Print transitions for all non-Sohncke groups
    anaddb-chiral --all
    
    # Print summary for all groups (use cache if available)
    anaddb-chiral --all --cache
    
    # Force refresh the cache
    anaddb-chiral --all --refresh-cache
    
    # Filter by daughter space group
    anaddb-chiral --sg 84 --daughter 77
    
    # Save output to file
    anaddb-chiral --sg 136 --output transitions.txt
"""

import argparse
import sys
from typing import Optional

from symphon.chiral_transitions import (
    ChiralTransitionFinder,
    format_transition_table,
    is_sohncke,
    get_sohncke_numbers,
    get_sohncke_class,
    SohnckeClass,
    _save_transitions_cache,
    _load_transitions_cache,
)


def print_parent_info(finder: ChiralTransitionFinder) -> None:
    """Print parent space group information."""
    info = finder.spacegroup_info
    print(f"Parent: {info.symbol} (#{info.number})")
    print(f"Order: {info.order}")
    print(f"Point group: {info.point_group_symbol}")
    print(f"Is centrosymmetric: {info.has_inversion()}")
    print(f"Sohncke class: {get_sohncke_class(info.number).value}")
    print()
    print("Symmetry Operations:")
    print(info.get_operations_report())
    print()


def print_transitions(
    finder: ChiralTransitionFinder,
    daughter_filter: Optional[int] = None,
    verbose: bool = False,
) -> int:
    """
    Find and print chiral transitions.
    
    Returns:
        Number of transitions found
    """
    try:
        from symphon.chiral_transitions import HAS_SPGREP, HAS_SPGREP_MODULATION
        if not HAS_SPGREP or not HAS_SPGREP_MODULATION:
            print("Error: 'spgrep' and 'spgrep-modulation' are required for chiral transition analysis.", file=sys.stderr)
            print("Install them with: pip install spgrep spgrep-modulation", file=sys.stderr)
            return 0
            
        transitions = finder.find_chiral_transitions()
    except Exception as e:
        print(f"Error finding transitions: {e}", file=sys.stderr)
        return 0
    
    if daughter_filter is not None:
        transitions = [t for t in transitions if t.daughter_spg_number == daughter_filter]
    
    if not transitions:
        print("No chiral transitions found.")
        return 0
    
    print(f"Found {len(transitions)} chiral transitions")
    print()
    
    if verbose:
        # Show unique daughters first
        daughters = {}
        for t in transitions:
            if t.daughter_spg_number not in daughters:
                daughters[t.daughter_spg_number] = []
            daughters[t.daughter_spg_number].append(t)
        
        print("Unique daughter space groups:")
        for spg, trans in sorted(daughters.items()):
            print(f"  {trans[0].daughter_spg_symbol} (#{spg}): {len(trans)} transitions")
        print()
    
    print(format_transition_table(transitions))
    return len(transitions)


def print_summary_for_all(use_cache: bool = True, refresh: bool = False) -> None:
    """Print summary table for all non-Sohncke space groups."""
    print("=" * 100)
    print("Chiral Transitions Summary - All Non-Sohncke Space Groups")
    print("(Method: enumerating all isotropy subgroups)")
    print("=" * 100)
    print()
    
    cached_data = None
    if use_cache and not refresh:
        cached_data = _load_transitions_cache()
        if cached_data:
            print("(Using cached data. Use --refresh-cache to recompute.)")
            print()
    
    if cached_data:
        print(f"{'Parent':>8} {'Symbol':>12} {'Order':>6} {'Daughters':>40} {'Transitions':>10}")
        print("-" * 100)
        
        total_parents = 0
        total_transitions = 0
        for entry in cached_data:
            print(f"{entry['spg_num']:>8} {entry['symbol']:>12} {entry['order']:>6} "
                  f"{entry['daughters_str']:>40} {entry['num_transitions']:>10}")
            total_parents += 1
            total_transitions += entry['num_transitions']
        
        print("-" * 100)
        print(f"Total: {total_parents} non-Sohncke parents, {total_transitions} total transitions")
        return
    
    sohncke_set = set(get_sohncke_numbers())
    
    print(f"{'Parent':>8} {'Symbol':>12} {'Order':>6} {'Daughters':>40} {'Transitions':>10}")
    print("-" * 100)
    
    total_parents = 0
    total_transitions = 0
    cache_data = []
    
    for spg_num in range(1, 231):
        if spg_num in sohncke_set:
            continue
        
        try:
            finder = ChiralTransitionFinder(spg_num)
            
            from symphon.chiral_transitions import HAS_SPGREP
            if HAS_SPGREP:
                transitions = finder.find_chiral_transitions()
            else:
                # Should not happen if dependencies are checked earlier, 
                # but we can't easily exit here without breaking the loop
                continue
            
            if not transitions:
                continue
            
            total_parents += 1
            daughters = sorted(set(t.daughter_spg_number for t in transitions))
            daughter_str = ", ".join(f"#{d}" for d in daughters[:5])
            if len(daughters) > 5:
                daughter_str += f" (+{len(daughters) - 5} more)"
            
            print(f"{spg_num:>8} {finder.spacegroup_info.symbol:>12} "
                  f"{finder.spacegroup_info.order:>6} {daughter_str:>40} {len(transitions):>10}")
            total_transitions += len(transitions)
            
            cache_data.append({
                'spg_num': spg_num,
                'symbol': finder.spacegroup_info.symbol,
                'order': finder.spacegroup_info.order,
                'daughters_str': daughter_str,
                'num_transitions': len(transitions),
            })
            
        except Exception:
            continue
    
    print("-" * 100)
    print(f"Total: {total_parents} non-Sohncke parents, {total_transitions} total transitions")
    
    if use_cache and cache_data:
        try:
            _save_transitions_cache(cache_data)
            print("\n(Results cached for future use.)")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Find chiral phase transitions from space groups",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --sg 136              Print transitions for SG 136 (P4_2/mnm)
  %(prog)s --sg 84 --daughter 77 Print transitions from SG 84 to SG 77
  %(prog)s --all                 Print summary for all non-Sohncke groups
  %(prog)s --list-daughters      List all possible chiral daughter groups
        """,
    )
    
    parser.add_argument(
        "--sg", "--space-group",
        type=int,
        help="Space group number (1-230) to find transitions FROM",
    )
    
    parser.add_argument(
        "--daughter",
        type=int,
        help="Filter transitions by daughter space group number",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print summary for all non-Sohncke space groups",
    )
    
    parser.add_argument(
        "--list-daughters",
        action="store_true",
        help="List all possible chiral daughter space groups",
    )
    
    parser.add_argument(
        "--list-sons",
        action="store_true",
        help="List all space groups that can transition to a chiral daughter",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file (default: stdout)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    parser.add_argument(
        "--cache",
        action="store_true",
        default=True,
        help="Use cached results for --all (default: True)",
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching for --all",
    )
    
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force refresh the cache for --all",
    )
    
    args = parser.parse_args()
    
    if args.no_cache:
        args.cache = False
    
    # Redirect output if requested
    if args.output:
        with open(args.output, "w") as f:
            old_stdout = sys.stdout
            sys.stdout = f
            try:
                main_inner(args, parser)
            finally:
                sys.stdout = old_stdout
        print(f"Output saved to {args.output}")
    else:
        main_inner(args, parser)


def main_inner(args, parser):
    """Inner main function."""
    if args.all:
        print_summary_for_all(
            use_cache=args.cache, 
            refresh=args.refresh_cache,
        )
        return
    
    if args.list_daughters:
        print("Possible chiral daughter space groups (Sohncke groups):")
        print()
        sohncke_numbers = get_sohncke_numbers()
        
        # Group by crystal system
        systems = {
            "Triclinic": [],
            "Monoclinic": [],
            "Orthorhombic": [],
            "Tetragonal": [],
            "Trigonal": [],
            "Hexagonal": [],
            "Cubic": [],
        }
        
        # Simple classification by number ranges
        for num in sohncke_numbers:
            if num <= 2:
                systems["Triclinic"].append(num)
            elif num <= 15:
                systems["Monoclinic"].append(num)
            elif num <= 74:
                systems["Orthorhombic"].append(num)
            elif num <= 142:
                systems["Tetragonal"].append(num)
            elif num <= 167:
                systems["Trigonal"].append(num)
            elif num <= 194:
                systems["Hexagonal"].append(num)
            else:
                systems["Cubic"].append(num)
        
        for system, nums in systems.items():
            if nums:
                print(f"{system}: {', '.join(f'#{n}' for n in nums)}")
        return
    
    if args.sg:
        if not 1 <= args.sg <= 230:
            print(f"Error: Invalid space group number {args.sg} (must be 1-230)", file=sys.stderr)
            sys.exit(1)
        
        if is_sohncke(args.sg):
            print(f"Space group {args.sg} is already a Sohncke group (chiral).")
            print("It cannot have chiral transitions FROM it.")
            print("Use it as a --daughter filter instead.")
            sys.exit(1)
        
        finder = ChiralTransitionFinder(args.sg)
        print_parent_info(finder)
        print_transitions(
            finder, 
            daughter_filter=args.daughter, 
            verbose=args.verbose,
        )
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
