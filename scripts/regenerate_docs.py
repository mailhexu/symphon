#!/usr/bin/env python
"""
Regenerate chiral transition tables for documentation.

Usage:
    python regenerate_docs.py [--start N] [--end N] [--output-dir DIR]

Generates .txt files for each non-Sohncke space group using the anaddb-chiral CLI.
"""

import subprocess
import sys
from pathlib import Path


def get_sohncke_numbers():
    """Get the set of Sohncke space group numbers."""
    from symphon.chiral_transitions import get_sohncke_numbers
    return set(get_sohncke_numbers())


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate chiral transition docs")
    parser.add_argument("--start", type=int, default=1, help="Start space group number")
    parser.add_argument("--end", type=int, default=230, help="End space group number")
    parser.add_argument("--output-dir", type=str, default="docs/chiral_transitions", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sohncke = get_sohncke_numbers()
    
    cli_path = Path(__file__).parent.parent / "anaddb_irreps" / "cli_chiral.py"
    
    for sg in range(args.start, args.end + 1):
        if sg in sohncke:
            continue
        
        output_file = output_dir / f"{sg}.txt"
        
        if output_file.exists() and not args.force:
            print(f"Skipping SG {sg} (file exists)")
            continue
        
        print(f"Processing SG {sg}...", end=" ", flush=True)
        
        try:
            result = subprocess.run(
                [sys.executable, str(cli_path), "--sg", str(sg)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                with open(output_file, "w") as f:
                    f.write(result.stdout)
                print(f"Done -> {output_file}")
            else:
                print(f"Error: {result.stderr[:100]}")
                
        except subprocess.TimeoutExpired:
            print("Timeout")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
