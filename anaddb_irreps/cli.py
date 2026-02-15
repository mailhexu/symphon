"""Command-line interface for anaddb_irreps.

Provides a concise irreps summary by default and optional verbose output.
"""

import argparse
from .irreps_anaddb import IrRepsAnaddb, IrRepsPhonopy, print_irreps_phonopy


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

    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        default="phonopy",
        choices=["phonopy", "irrep"],
        help="Backend driver to use for irrep identification (default: phonopy)",
    )
    parser.add_argument(
        "-k",
        "--kpname",
        type=str,
        default=None,
        help="k-point name (e.g. GM, X, M) used by 'irrep' backend",
    )
    parser.add_argument(
        "--both-labels",
        action="store_true",
        help="For Gamma point: show both phonopy (Mulliken) and irrep (BCS) labels",
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
        backend=args.backend,
        both_labels=args.both_labels,
    )
    irr.run(kpname=args.kpname)

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


def parse_args_phonopy() -> argparse.Namespace:
    """Parse command-line arguments for the phonopy-irreps CLI."""
    parser = argparse.ArgumentParser(
        prog="phonopy-irreps",
        description=(
            "Compute irreducible representations of phonon modes from phonopy "
            "params/YAML output. By default, analyzes all high-symmetry k-points "
            "automatically using the irrep backend."
        ),
    )

    parser.add_argument(
        "-p",
        "--params",
        dest="phonopy_params",
        required=True,
        help=(
            "Path to phonopy params/YAML file (e.g. phonopy_params.yaml or "
            "phonopy.yaml)"
        ),
    )

    parser.add_argument(
        "-s",
        "--symprec",
        type=float,
        default=None,
        help=(
            "Override symmetry precision used for structure analysis. "
            "If omitted, anaddb_irreps will try to use the "
            "symmetry tolerance stored in the phonopy file, "
            "falling back to 1e-5."
        ),
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
        help="Use little co-group setting (passes True to IrRepsPhonopy)",
    )
    parser.add_argument(
        "-v",
        "--log-level",
        type=int,
        default=0,
        help="Verbosity level passed to IrRepsPhonopy (default: 0)",
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


def main_phonopy() -> None:
    """Entry point for phonopy-irreps CLI.
    
    Automatically discovers and analyzes all high-symmetry k-points using the
    irrep backend. At Gamma point, shows both Mulliken and BCS labels.
    """
    from irrep.spacegroup_irreps import SpaceGroupIrreps
    from irreptables import IrrepTable
    from phonopy import load as phonopy_load
    
    args = parse_args_phonopy()
    
    # Load phonopy structure to get space group
    phonon = phonopy_load(args.phonopy_params)
    cell = phonon.primitive.cell
    positions = phonon.primitive.scaled_positions
    numbers = phonon.primitive.numbers
    
    # Determine symmetry precision: user input > yaml file > default
    if args.symprec is not None:
        symprec = args.symprec
    else:
        import yaml
        symprec = 1e-5  # default
        try:
            with open(args.phonopy_params, 'r') as f:
                yaml_data = yaml.safe_load(f)
            if yaml_data and 'phonopy' in yaml_data and 'symmetry_tolerance' in yaml_data['phonopy']:
                symprec = float(yaml_data['phonopy']['symmetry_tolerance'])
        except (IOError, KeyError, TypeError, ValueError):
            pass
    
    # Create SpaceGroupIrreps to get space group info
    sg = SpaceGroupIrreps.from_cell(
        cell=(cell, positions, numbers),
        spinor=False,
        include_TR=False,
        search_cell=True,
        symprec=symprec,
        verbosity=args.log_level
    )
    
    # Get all high-symmetry k-points from irrep package
    irrep_table = IrrepTable(sg.number_str, False, v=args.log_level)
    
    # Collect unique high-symmetry points
    high_sym_points = {}
    for irrep in irrep_table.irreps:
        if hasattr(irrep, 'kpname') and irrep.kpname and hasattr(irrep, 'k'):
            kpname = irrep.kpname
            k = tuple(irrep.k.tolist())
            if kpname not in high_sym_points:
                high_sym_points[kpname] = k
    
    # Print header with space group info (once at the top)
    print(f"Space group: {sg.name}")
    print(f"Found {len(high_sym_points)} high-symmetry points:")
    for kpname in sorted(high_sym_points.keys()):
        k = high_sym_points[kpname]
        print(f"  {kpname}: k={k}")
    print()
    
    # Analyze each high-symmetry point
    for kpname in sorted(high_sym_points.keys()):
        k = high_sym_points[kpname]
        print(f"# {kpname} point: k = [{k[0]:.4f}, {k[1]:.4f}, {k[2]:.4f}]")
        print("=" * 60)
        
        # At Gamma, use both labels; otherwise just irrep backend
        is_gamma = all(abs(x) < 1e-6 for x in k)
        
        irr = IrRepsPhonopy(
            phonopy_params=args.phonopy_params,
            qpoint=k,
            is_little_cogroup=args.is_little_cogroup,
            symprec=symprec,
            degeneracy_tolerance=args.degeneracy_tolerance,
            log_level=args.log_level,
            backend="irrep",
            both_labels=is_gamma,  # Dual labels only at Gamma
        )
        irr.run(kpname=kpname)
        print(irr.format_summary_table(include_symmetry=False, include_qpoint_cols=False))
        print()  # Add blank line between k-points
        
        # Optional verbose output
        if args.show_verbose or args.verbose_file:
            verbose_text = irr.get_verbose_output()
            
            if args.verbose_file:
                # Append to file for each k-point
                mode = 'a' if kpname != sorted(high_sym_points.keys())[0] else 'w'
                with open(args.verbose_file, mode, encoding="utf-8") as fh:
                    fh.write(f"\n# {kpname} point\n")
                    fh.write(verbose_text)
            elif args.show_verbose:
                print()
                print(f"# Verbose output for {kpname}")
                print(verbose_text, end="")


if __name__ == "__main__":  # pragma: no cover
    main()
