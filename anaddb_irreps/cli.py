"""Command-line interface for anaddb_irreps.

This script provides a simple CLI wrapper around ``IrRepsAnaddb`` so that
users can obtain irreducible representations directly from a PHBST file
without writing a Python script.

Usage:
    uv run python -m anaddb_irreps.cli --phbst run_PHBST.nc --q-index 0

The CLI roughly mirrors the ``IrRepsAnaddb`` constructor arguments and
prints the table produced by ``IrReps.run()`` / ``_show`` to stdout.
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

    # IrReps base class exposes _show; follow README example to keep behavior.
    # Use the same "show_all" style as example (True).
    try:
        irr._show(True)  # type: ignore[attr-defined]
    except AttributeError:
        # Fallback: just print the raw irreps object repr.
        print(irr)


if __name__ == "__main__":  # pragma: no cover
    main()
