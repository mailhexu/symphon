"""Unified symphon command-line interface with subcommands.

Usage:
    symphon anaddb-irreps         --phbst run_PHBST.nc --q-index 0
    symphon phonopy-irreps        --params phonopy_params.yaml
    symphon find-chiral-transition --sg 136
    symphon magnetic-chiral       --structure POSCAR --qpoint 0 0 0.5 --mag-sites 0,1
    symphon abstract-magnetic     --spg 221
    symphon msg                   18.16
"""

import sys
import argparse


def main() -> None:
    """Entry point for the unified symphon CLI."""
    parser = argparse.ArgumentParser(
        prog="symphon",
        description="symphon: phonon irreps and chiral transition tools.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="subcommand")
    subparsers.required = True

    # Register subcommands (lazily import their parsers to avoid heavy imports
    # at startup when only --help is requested).
    subparsers.add_parser("anaddb-irreps", help="Irreps from anaddb PHBST file")
    subparsers.add_parser("phonopy-irreps", help="Irreps from phonopy params/YAML file")
    subparsers.add_parser("find-chiral-transition", help="Find chiral phase transitions")
    subparsers.add_parser("magnetic-chiral", help="Find chiral magnetic transitions")
    subparsers.add_parser("abstract-magnetic", help="Abstract chiral magnetic transitions by space group")
    subparsers.add_parser("msg", help="Identify chirality of a Magnetic Space Group")
    subparsers.add_parser("modulate", help="Generate modulated supercells using high-symmetry basis")

    # Parse only the subcommand name; let each sub-CLI re-parse sys.argv[2:].
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    subcommand = sys.argv[1]

    # Rewrite sys.argv so sub-CLIs see their own prog name and arguments.
    if subcommand == "anaddb-irreps":
        from symphon.cli.main import main as _main
        sys.argv = ["symphon anaddb-irreps"] + sys.argv[2:]
        _main()

    elif subcommand == "phonopy-irreps":
        from symphon.cli.main import main_phonopy as _main
        sys.argv = ["symphon phonopy-irreps"] + sys.argv[2:]
        _main()

    elif subcommand == "find-chiral-transition":
        from symphon.cli_chiral import main as _main
        sys.argv = ["symphon find-chiral-transition"] + sys.argv[2:]
        _main()

    elif subcommand == "magnetic-chiral":
        from symphon.cli_magnetic import main as _main
        sys.argv = ["symphon magnetic-chiral"] + sys.argv[2:]
        _main()

    elif subcommand == "abstract-magnetic":
        from symphon.cli_abstract_magnetic import main as _main
        sys.argv = ["symphon abstract-magnetic"] + sys.argv[2:]
        _main()

    elif subcommand == "msg":
        from symphon.cli_msg_chiral import main as _main
        sys.argv = ["symphon msg"] + sys.argv[2:]
        _main()

    elif subcommand == "modulate":
        from symphon.cli_modulate import main as _main
        sys.argv = ["symphon modulate"] + sys.argv[2:]
        _main()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
