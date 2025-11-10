"""Command-line interface for anaddb_irreps.

Provides a concise irreps summary by default and optional verbose output.
"""

import argparse
from .irreps_anaddb import IrRepsAnaddb


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the anaddb_irreps CLI."""
    parser = argparse.ArgumentParser(
        prog="anaddb-irreps",
        description=(
            "Compute irreducible representations of phonon modes from anaddb "
            "PHBST output using anaddb_irreps (phonopy wrapper)."
        ),
    )

    parser.add_argument(
        "-p",
        "--phbst",
        dest="phbst_fname",
        required=True,
        help="Path to PHBST NetCDF file (e.g. run_PHBST.nc)",
    )
    parser.add_argument(
        "-q",
        "--q-index",
        dest="ind_q",
        type=int,
        required=True,
        help="Index of q-point in PHBST file (0-based)",
    )
    parser.add_argument(
        "-s",
        "--symprec",
        type=float,
        default=1e-5,
        help="Symmetry precision used by phonopy (default: 1e-5)",
    )
    parser.add_argument(
        "-d",
        "--degeneracy-tolerance",
        type=float,
        default=1e-4,
        help=(
            "Frequency difference tolerance for degeneracy detection "
            "(default: 1e-4)"
        ),
    )
    parser.add_argument(
        "-l",
        "--is-little-cogroup",
        action="store_true",
        help="Use little co-group setting (passes True to IrRepsAnaddb)",
    )
    parser.add_argument(
        "-v",
        "--log-level",
        type=int,
        default=0,
        help="Verbosity level passed to IrRepsAnaddb (default: 0)",
    )
    parser.add_argument(
        "--show-verbose",
        action="store_true",
        help=(
            "Also print the full verbose irreps output (phonopy-style). "
            "By default only a concise table is shown."
        ),
    )
    parser.add_argument(
        "--verbose-file",
        type=str,
        default=None,
        help=(
            "If set, write the verbose irreps output to this file instead of "
            "stdout (only used when --show-verbose is given)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for the anaddb_irreps CLI."""
    args = parse_args()

    irr = IrRepsAnaddb(
        phbst_fname=args.phbst_fname,
        ind_q=args.ind_q,
        is_little_cogroup=args.is_little_cogroup,
        symprec=args.symprec,
        degeneracy_tolerance=args.degeneracy_tolerance,
        log_level=args.log_level,
    )
    irr.run()

    # 1) Always show concise summary table (including IR/Raman activity)
    print(irr.format_summary_table())

    # 2) Optional verbose output
    if args.show_verbose or args.verbose_file:
        verbose_text = irr.get_verbose_output()

        if args.verbose_file:
            with open(args.verbose_file, "w", encoding="utf-8") as fh:
                fh.write(verbose_text)
        elif args.show_verbose:
            print()
            print("# Verbose irreps output")
            print(verbose_text, end="")


if __name__ == "__main__":  # pragma: no cover
    main()
