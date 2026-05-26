import lzma
import zipfile
from pathlib import Path

import pytest

from symphon.irreps_anaddb import IrRepsAnaddb, IrRepsPhonopy, find_highsym_qpoints_in_phbst
from symphon.full_star_isotropy import BcsFullStarContext


PROJECT_ROOT = Path(__file__).parent.parent
DEBUG_DIR = PROJECT_ROOT / "debug"


def _phonopy_yaml_from_zip(name: str, tmp_path: Path) -> Path:
    zip_path = DEBUG_DIR / f"{name}.zip"
    if not zip_path.exists():
        pytest.skip(f"Debug fixture not available: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        payload = zf.read("phonopy_params.yaml.xz")

    yaml_path = tmp_path / name / "phonopy_params.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_bytes(lzma.decompress(payload))
    return yaml_path


def _assert_no_missing_bcs_labels(irr):
    labels = irr.get_bcs_labels()
    assert labels
    assert all(label not in (None, "-") for label in labels)


def test_debug_sr2hfo4_x_and_m_have_labels(tmp_path):
    yaml_path = _phonopy_yaml_from_zip("Sr2HfO4", tmp_path)

    for kpname, qpoint in [("X", [0, 0, 0.5]), ("M", [0.5, 0.5, 0.5])]:
        irr = IrRepsPhonopy(str(yaml_path), qpoint=qpoint)
        irr.run(kpname=kpname)
        _assert_no_missing_bcs_labels(irr)


def test_debug_sr2hfo4_x_full_star_context_finds_class_ii_daughter_set(tmp_path):
    yaml_path = _phonopy_yaml_from_zip("Sr2HfO4", tmp_path)

    irr = IrRepsPhonopy(str(yaml_path), qpoint=[0, 0, 0.5])
    irr.run(kpname="X")
    backend = irr._irrep_backend_obj

    daughters = {}
    for label in ["X1", "X2"]:
        ref, little_indices = backend._get_reference_matrices(backend._sg_obj, [], label)
        ctx = BcsFullStarContext.from_spacegroup_irrep(
            backend._sg_obj,
            backend._qpoint,
            little_indices,
            ref,
            backend._symprec,
        )
        matrices, arms, _reps = ctx.induced_matrices()
        daughter_numbers = set()
        for opd in ([1, 0, 1, 0], [0, 1, 0, 1]):
            number, _symbol, _order = ctx.identify_full_star_daughter(
                opd,
                backend._primitive.cell,
                matrices=matrices,
                arms=arms,
            )
            daughter_numbers.add(number)
        daughters[label] = daughter_numbers

    assert daughters["X1"] == {91, 95}
    assert daughters["X2"] == {92, 96}


def test_debug_sr2hfo4_x_chiral_reporting_prefers_full_star_daughters(tmp_path):
    yaml_path = _phonopy_yaml_from_zip("Sr2HfO4", tmp_path)

    irr = IrRepsPhonopy(str(yaml_path), qpoint=[0, 0, 0.5])
    irr._compute_chiral = True
    irr.run(kpname="X")

    daughters = {row["daughter_sg"] for row in irr.get_summary_table()}
    assert any("#91" in daughter for daughter in daughters)
    assert any("#95" in daughter for daughter in daughters)
    assert any("#92" in daughter for daughter in daughters)
    assert any("#96" in daughter for daughter in daughters)

    formatted = irr.format_summary_table(show_chiral=True)
    assert "II-pair" in formatted


def test_debug_sr2hfo4_x_full_star_opds_without_chiral_flag(tmp_path):
    yaml_path = _phonopy_yaml_from_zip("Sr2HfO4", tmp_path)

    irr = IrRepsPhonopy(str(yaml_path), qpoint=[0, 0, 0.5])
    irr.run(kpname="X")

    summary = irr.get_summary_table()
    assert summary[0]["opd"] == "(a, 0, a, 0)"
    assert summary[1]["opd"] == "(0, a, 0, a)"

    daughters = {row["daughter_sg"] for row in summary}
    assert any("#91" in daughter for daughter in daughters)
    assert any("#95" in daughter for daughter in daughters)
    assert any("#92" in daughter for daughter in daughters)
    assert any("#96" in daughter for daughter in daughters)


def test_debug_bage2s5_w_has_labels(tmp_path):
    yaml_path = _phonopy_yaml_from_zip("BaGe2S5", tmp_path)

    irr = IrRepsPhonopy(str(yaml_path), qpoint=[0.5, 0.25, 0.75])
    irr.run(kpname="W")
    _assert_no_missing_bcs_labels(irr)


def test_debug_bage2s5_w_chiral_output_has_primitive_opds(tmp_path):
    yaml_path = _phonopy_yaml_from_zip("BaGe2S5", tmp_path)

    irr = IrRepsPhonopy(str(yaml_path), qpoint=[0.5, 0.25, 0.75])
    irr._compute_chiral = True
    irr.run(kpname="W")

    summary = irr.get_summary_table()
    assert any(row["opd"] in {"(a, 0)", "(0, a)"} for row in summary)
    assert any(row["opd_prim"] in {"(a, 0)", "(0, a)"} for row in summary)
    assert any(row["daughter_sg"] != "-" for row in summary)

    formatted = irr.format_summary_table(show_chiral=True)
    assert "OPD(BCS)" in formatted
    assert "OPD(prim)" in formatted
    assert "(a, 0)" in formatted


def test_debug_bage2s5_full_star_domain_table_reports_x1_and_w_domains(tmp_path):
    yaml_path = _phonopy_yaml_from_zip("BaGe2S5", tmp_path)

    irr_x = IrRepsPhonopy(str(yaml_path), qpoint=[0.5, 0.0, 0.5])
    irr_x.run(kpname="X")
    x_rows = irr_x.get_full_star_domain_table()

    x1_daughters = {row["daughter_sg"] for row in x_rows if row["label"] == "X1"}
    x1_chiral = {row["chiral"] for row in x_rows if row["label"] == "X1"}
    assert any("#91" in daughter for daughter in x1_daughters)
    assert any("#95" in daughter for daughter in x1_daughters)
    assert "II-pair" in x1_chiral

    irr_w = IrRepsPhonopy(str(yaml_path), qpoint=[0.5, 0.25, 0.75])
    irr_w.run(kpname="W")
    w_rows = irr_w.get_full_star_domain_table()

    assert any(row["label"] == "W1" for row in w_rows)
    assert any(row["label"] == "W2" for row in w_rows)
    assert any("sqrt(2)" in row["opd"] for row in w_rows if row["label"] in {"W1", "W2"})


def test_debug_sio2_x_primitive_opd_has_full_star_coordinates(tmp_path):
    yaml_path = _phonopy_yaml_from_zip("SiO2", tmp_path)

    irr = IrRepsPhonopy(str(yaml_path), qpoint=[0, 0, 0.5])
    irr._compute_chiral = True
    irr.run(kpname="X")

    assert irr._irrep_opds_prim[:2] == ["(a, 0, a, 0)", "(0, a, 0, a)"]


def test_debug_sio2_m3_daughter_uses_snapped_highsym_qpoint(tmp_path):
    yaml_path = _phonopy_yaml_from_zip("SiO2", tmp_path)

    irr = IrRepsPhonopy(
        str(yaml_path),
        qpoint=[0.5000000000000001, 0.5000000000000001, 0.4999999999999999],
    )
    irr.run(kpname="M")

    m3_daughters = {row["daughter_sg"] for row in irr.get_summary_table()[8:12]}

    assert any("#91" in daughter for daughter in m3_daughters)
    assert any("#95" in daughter for daughter in m3_daughters)


def test_debug_zrsi_phbst_has_labels():
    phbst_path = DEBUG_DIR / "ZrSi_444_anaddbo_PHBST.nc"
    if not phbst_path.exists():
        pytest.skip(f"Debug fixture not available: {phbst_path}")

    matched = find_highsym_qpoints_in_phbst(str(phbst_path))
    wanted = {"GM", "M", "N", "P", "X"}
    by_label = {entry["label"]: entry for entry in matched if entry["label"] in wanted}
    assert wanted <= set(by_label)

    for kpname in sorted(wanted):
        entry = by_label[kpname]
        irr = IrRepsAnaddb(str(phbst_path), entry["ind_q"])
        irr.run(kpname=kpname)
        _assert_no_missing_bcs_labels(irr)
