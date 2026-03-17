"""Command-line interface for symphon.

Provides a concise irreps summary by default and optional verbose output.
"""

import argparse
import numpy as np
from .irreps_anaddb import IrRepsAnaddb, IrRepsPhonopy, find_highsym_qpoints_in_phbst, print_irreps_phonopy


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the symphon CLI."""
    parser = argparse.ArgumentParser(
        prog="anaddb-irreps",
        description=(
            "Compute irreducible representations of phonon modes from anaddb "
            "PHBST output using symphon (phonopy wrapper)."
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
        default=None,
        help=(
            "Index of q-point in PHBST file (0-based). "
            "If omitted, all high-symmetry q-points found in the file are analyzed."
        ),
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
        "-k",
        "--kpname",
        type=str,
        default=None,
        help=(
            "k-point name (e.g. GM, X, M). Used only with -q; "
            "in auto-scan mode the label is determined automatically."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the anaddb-irreps CLI."""
    args = parse_args()

    if args.ind_q is not None:
        # ── Single q-point mode ──────────────────────────────────────────────
        irr = IrRepsAnaddb(
            phbst_fname=args.phbst_fname,
            ind_q=args.ind_q,
            symprec=args.symprec,
            degeneracy_tolerance=args.degeneracy_tolerance,
            log_level=args.log_level,
        )
        irr.run(kpname=args.kpname)
        print(irr.format_summary_table())

        if args.show_verbose or args.verbose_file:
            verbose_text = irr.get_verbose_output()
            if args.verbose_file:
                with open(args.verbose_file, "w", encoding="utf-8") as fh:
                    fh.write(verbose_text)
            elif args.show_verbose:
                print()
                print("# Verbose irreps output")
                print(verbose_text, end="")

    else:
        # ── Auto high-symmetry scan mode ─────────────────────────────────────
        matched = find_highsym_qpoints_in_phbst(
            args.phbst_fname,
            symprec=args.symprec,
        )

        if not matched:
            print(
                "No high-symmetry q-points found in the PHBST file. "
                "Use -q to specify a q-point index explicitly."
            )
            return

        print(f"Found {len(matched)} high-symmetry q-point(s) in {args.phbst_fname}:")
        for m in matched:
            q = m["qpoint"]
            print(f"  [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}]  label={m['label']}  ind_q={m['ind_q']}")
        print()

        verbose_file_handle = None
        if args.verbose_file:
            verbose_file_handle = open(args.verbose_file, "w", encoding="utf-8")

        try:
            for m in matched:
                irr = IrRepsAnaddb(
                    phbst_fname=args.phbst_fname,
                    ind_q=m["ind_q"],
                    symprec=args.symprec,
                    degeneracy_tolerance=args.degeneracy_tolerance,
                    log_level=args.log_level,
                )
                irr.run(kpname=m["label"])

                q = m["qpoint"]
                print(f"# {m['label']} point  (ind_q={m['ind_q']})")
                print(f"# k = [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}]")
                print("=" * 60)
                print(irr.format_summary_table(include_symmetry=False, include_qpoint_cols=False))
                print()

                if args.show_verbose or args.verbose_file:
                    verbose_text = irr.get_verbose_output()
                    if verbose_file_handle:
                        verbose_file_handle.write(f"# {m['label']} point\n")
                        verbose_file_handle.write(verbose_text)
                    elif args.show_verbose:
                        print(f"# Verbose output for {m['label']}")
                        print(verbose_text, end="")
                        print()
        finally:
            if verbose_file_handle:
                verbose_file_handle.close()


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
            "If omitted, symphon will try to use the "
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
    parser.add_argument(
        "--chiral",
        action="store_true",
        help=(
            "Compute and display possible chiral symmetry-breaking transitions "
            "for each mode (OPD and daughter space group columns)."
        ),
    )
    parser.add_argument(
        "--compare-ground-truth",
        action="store_true",
        dest="compare_ground_truth",
        help=(
            "Compare daughter space groups with ground truth from spgrep-modulation. "
            "Shows both implementation and ground truth results side-by-side."
        ),
    )

    return parser.parse_args()


def main_phonopy() -> None:
    """Entry point for phonopy-irreps CLI.
    
    Automatically discovers and analyzes all high-symmetry k-points using the
    irrep backend. At Gamma point, shows both Mulliken and BCS labels.
    """
    from irrep.spacegroup_irreps import SpaceGroupIrreps
    try:
        from irreptables.irreps import IrrepTable
    except ImportError:
        from irreptables import IrrepTable  # type: ignore
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
    # Skip k-points that are equivalent (mod 1) to already-seen ones
    # Transform BCS k-points to primitive coordinates using refUC
    refUCinv = np.linalg.inv(sg.refUC.T)
    high_sym_points = {}
    seen_k_equiv = []
    for irrep in irrep_table.irreps:
        if hasattr(irrep, 'kpname') and irrep.kpname and hasattr(irrep, 'k'):
            kpname = irrep.kpname
            k_bcs = np.array(irrep.k.tolist())
            k_prim = refUCinv @ k_bcs
            k_equiv = tuple(np.round(k_prim - np.round(k_prim), 6))
            if kpname not in high_sym_points and k_equiv not in seen_k_equiv:
                high_sym_points[kpname] = tuple(k_prim.tolist())
                seen_k_equiv.append(k_equiv)
    
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
        
        # At Gamma, use both labels; otherwise just irrep backend
        is_gamma = all(abs(x) < 1e-6 for x in k)
        
        irr = IrRepsPhonopy(
            phonopy_params=args.phonopy_params,
            qpoint=k,
            #is_little_cogroup=args.is_little_cogroup,
            symprec=symprec,
            degeneracy_tolerance=args.degeneracy_tolerance,
            log_level=args.log_level,
            #backend="irrep",
            #both_labels=is_gamma,  # Dual labels only at Gamma
        )
        irr._compute_chiral = args.chiral
        irr.run(kpname=kpname)
        
        # Print k-point header with coordinate mapping
        print(f"# {kpname} point (Primitive coordinates)")
        print(f"# k_prim = [{k[0]:.4f}, {k[1]:.4f}, {k[2]:.4f}]")
        
        # Get BCS coordinates and label if available
        backend_obj = getattr(irr, '_irrep_backend_obj', None)
        if backend_obj and hasattr(backend_obj, '_qpoint_bcs') and hasattr(backend_obj, '_bcs_kpname'):
            q_bcs = backend_obj._qpoint_bcs
            bcs_label = backend_obj._bcs_kpname
            print(f"# k_BCS  = [{q_bcs[0]:.4f}, {q_bcs[1]:.4f}, {q_bcs[2]:.4f}]  (BCS label: {bcs_label})")
        
        print("=" * 60)
        
        # Compute ground truth if requested
        ground_truth_data = None
        if args.compare_ground_truth:
            ground_truth_data = compute_ground_truth_daughters(
                phonon, k, symprec
            )
        
        print(irr.format_summary_table(
            include_symmetry=False, 
            include_qpoint_cols=False, 
            show_chiral=args.chiral,
            ground_truth=ground_truth_data
        ))
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


def compute_ground_truth_daughters(phonon, qpoint, symprec=1e-5):
    """
    Compute ground truth daughter space groups using spgrep-modulation.

    Returns a flat list of (freq, daughter_sg) — one entry per OPD direction
    (i.e. one per band), sorted by frequency.  Degenerate eigenspaces of
    dimension d contribute d consecutive entries at the same frequency, one
    for each standard-basis OPD vector j=0…d-1.

    Parameters
    ----------
    phonon : Phonopy
    qpoint : tuple
    symprec : float

    Returns
    -------
    list of (float, str)  or  None on failure
    """
    try:
        from spgrep_modulation.modulation import Modulation
        import numpy as np
        import spglib
    except ImportError:
        return None

    try:
        qpoint_arr = np.array(qpoint)

        # Determine supercell matrix from q-point
        denoms = [1, 1, 1]
        for i, x in enumerate(qpoint_arr):
            if np.isclose(x, 0, atol=1e-5):
                continue
            for d in range(1, 13):
                if np.isclose((x * d) % 1.0, 0, atol=1e-5):
                    denoms[i] = d
                    break
        supercell_matrix = np.diag(denoms)

        md = Modulation.with_supercell_and_symmetry_search(
            dynamical_matrix=phonon.dynamical_matrix,
            supercell_matrix=supercell_matrix,
            qpoint=qpoint_arr,
            factor=phonon.unit_conversion_factor,
            symprec=symprec,
        )

        # Build flat list: one (freq, daughter_sg) per OPD direction
        flat = []
        for i, (eigval, eigvecs, irrep) in enumerate(md.eigenspaces):
            freq = md.eigvals_to_frequencies(eigval)
            dim = eigvecs.shape[0]

            for j in range(dim):
                opd = np.zeros(dim, dtype=complex)
                opd[j] = 1.0
                amplitudes = list(np.abs(opd) * 0.1)
                arguments = list(np.angle(opd))

                daughter = "-"
                try:
                    cell, mod = md.get_modulated_supercell_and_modulation(
                        frequency_index=i,
                        amplitudes=amplitudes,
                        arguments=arguments,
                        return_cell=True,
                    )
                    dataset = spglib.get_symmetry_dataset(
                        (cell.cell, cell.scaled_positions, cell.numbers),
                        symprec=symprec,
                    )
                    if dataset is not None:
                        daughter = f"{dataset.international}(#{dataset.number})"
                except Exception:
                    pass

                flat.append((float(freq), daughter))

        # Sort by frequency so alignment by position is robust
        flat.sort(key=lambda x: x[0])
        return flat
    except Exception:
        return None


if __name__ == "__main__":  # pragma: no cover
    main()
