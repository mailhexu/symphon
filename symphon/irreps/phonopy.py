import numpy as np
from phonopy import load as phonopy_load
from .core import IrRepsEigen

class IrRepsPhonopy(IrRepsEigen):
    """Irreps helper for direct phonopy calculations."""

    def __init__(
        self,
        phonopy_params,
        qpoint,
        symprec: float | None = None,
        degeneracy_tolerance: float = 1e-5,
        log_level: int = 0,
    ) -> None:
        phonon = phonopy_load(phonopy_params)
        q = np.asarray(qpoint, dtype=float)
        phonon.run_qpoints([q], with_eigenvectors=True)
        q_dict = phonon.get_qpoints_dict()
        freqs = np.array(q_dict["frequencies"][0], dtype=float)
        eigvecs = np.array(q_dict["eigenvectors"][0], dtype=complex)
        primitive_atoms = phonon.primitive

        if symprec is None:
            symprec = phonon._symprec

        super().__init__(
            primitive_atoms,
            qpoint,
            freqs,
            eigvecs,
            symprec=symprec,
            degeneracy_tolerance=degeneracy_tolerance,
            log_level=log_level,
        )


def print_irreps_phonopy(
    phonopy_params,
    qpoint,
    symprec: float | None = None,
    degeneracy_tolerance: float = 1e-4,
    log_level: int = 0,
    show_verbose: bool = False,
    kpname=None,
):
    irr = IrRepsPhonopy(
        phonopy_params=phonopy_params,
        qpoint=qpoint,
        symprec=symprec,
        degeneracy_tolerance=degeneracy_tolerance,
        log_level=log_level,
    )
    irr.run(kpname=kpname)

    # Print summary table
    print(irr.format_summary_table())

    # Optionally print verbose output
    if show_verbose:
        print()
        print("# Verbose irreps output")
        print(irr.get_verbose_output())

    return irr
