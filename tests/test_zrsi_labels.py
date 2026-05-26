from pathlib import Path

import pytest

from symphon.irreps_anaddb import IrRepsPhonopy


PROJECT_ROOT = Path(__file__).parent.parent
ZRSI_PARAMS = PROJECT_ROOT / "examples" / "ZrSi" / "phonopy_params.yaml"


@pytest.mark.parametrize(
    ("kpname", "qpoint"),
    [
        ("GM", [0, 0, 0]),
        ("M", [0.5, 0.5, 0.5]),
        ("N", [0, 0.5, 0]),
        ("P", [0.25, 0.25, 0.25]),
        ("X", [0, 0, 0.5]),
    ],
)
def test_zrsi_high_symmetry_points_have_bcs_labels(kpname, qpoint):
    if not ZRSI_PARAMS.exists():
        pytest.skip("ZrSi example is not available")

    irr = IrRepsPhonopy(str(ZRSI_PARAMS), qpoint=qpoint)
    irr.run(kpname=kpname)

    labels = irr.get_bcs_labels()
    assert labels
    assert all(label not in (None, "-") for label in labels)


def test_zrsi_x_acoustic_modes_have_opds():
    if not ZRSI_PARAMS.exists():
        pytest.skip("ZrSi example is not available")

    irr = IrRepsPhonopy(str(ZRSI_PARAMS), qpoint=[0, 0, 0.5])
    irr._compute_chiral = True
    irr.run(kpname="X")

    opds = irr._irrep_opds_bcs
    assert opds[:2]
    assert all(opd not in (None, "-") for opd in opds[:2])


def test_zrsi_x_reports_bcs_and_primitive_opds():
    if not ZRSI_PARAMS.exists():
        pytest.skip("ZrSi example is not available")

    irr = IrRepsPhonopy(str(ZRSI_PARAMS), qpoint=[0, 0, 0.5])
    irr._compute_chiral = True
    irr.run(kpname="X")

    assert all(opd not in (None, "-") for opd in irr._irrep_opds_bcs[:2])
    assert all(opd not in (None, "-") for opd in irr._irrep_opds_prim[:2])
    assert all(dsg not in (None, "-") for dsg in irr._irrep_daughters_prim[:2])
    assert irr._irrep_opds_prim[0] == "(a, 0, a, 0)"
    assert irr._irrep_opds_prim[1] == "(0, a, 0, a)"

    output = irr.format_summary_table(
        show_chiral=True,
        include_symmetry=False,
        include_qpoint_cols=False,
    )
    assert "OPD(BCS)" in output
    assert "OPD(prim)" in output
    assert "Daughter SG(BCS)" not in output
    assert "Daughter SG(prim)" not in output
    assert "Daughter SG" in output
