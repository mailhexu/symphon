from anaddb_irreps import IrRepsAnaddb


def show_phbst_irreps():
    irr = IrRepsAnaddb(
        phbst_fname="run_PHBST.nc",
        ind_q=0,
        symprec=1e-8,
        degeneracy_tolerance=1e-4)
    irr.run()
    irr._show(True)


show_phbst_irreps()
