"""
Test CLI output against reference outputs.

Run:
    pytest tests/test_cli.py -v
    
Generate references:
    pytest tests/test_cli.py --generate-references
"""

import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
REFERENCE_DIR = PROJECT_ROOT / "tests" / "reference"


def run_phonopy_irreps(yaml_file: Path) -> str:
    """Run phonopy-irreps on a YAML file and return output."""
    import sys
    result = subprocess.run(
        [sys.executable, "-c", 
         f"from anaddb_irreps.cli import main_phonopy; "
         f"import sys; sys.argv = ['phonopy-irreps', '-p', '{yaml_file}']; "
         f"main_phonopy()"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=120,
    )
    return result.stdout + result.stderr


def get_reference_path(yaml_file: Path) -> Path:
    """Get reference file path for a YAML file."""
    rel_path = yaml_file.relative_to(EXAMPLES_DIR)
    ref_name = rel_path.with_suffix(".txt").name
    return REFERENCE_DIR / rel_path.parent / ref_name


def normalize_output(output: str) -> str:
    """Normalize output for comparison (remove timing, paths, etc.)."""
    lines = []
    for line in output.splitlines():
        # Skip empty lines at start
        if not lines and not line.strip():
            continue
        lines.append(line)
    return "\n".join(lines).strip()


@pytest.mark.parametrize("yaml_file", [
    "2_basic_phonopy/BaTiO3_phonopy_params.yaml",
    "3_advanced_phonopy/TmFeO3_phonopy_params.yaml",
])
def test_phonopy_irreps_output(yaml_file: str, request):
    """Test that phonopy-irreps output matches reference."""
    yaml_path = EXAMPLES_DIR / yaml_file
    ref_path = get_reference_path(yaml_path)
    
    if not yaml_path.exists():
        pytest.skip(f"Example file not found: {yaml_path}")
    
    # Generate references mode
    if request.config.getoption("--generate-references"):
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        output = run_phonopy_irreps(yaml_path)
        ref_path.write_text(output)
        pytest.skip(f"Generated reference: {ref_path}")
    
    # Normal test mode
    if not ref_path.exists():
        pytest.skip(f"Reference not found: {ref_path}. Run with --generate-references to create.")
    
    output = run_phonopy_irreps(yaml_path)
    reference = ref_path.read_text()
    
    output_norm = normalize_output(output)
    reference_norm = normalize_output(reference)
    
    if output_norm != reference_norm:
        # Show diff for debugging
        diff_file = PROJECT_ROOT / "test_results" / f"{yaml_path.stem}_diff.txt"
        diff_file.parent.mkdir(parents=True, exist_ok=True)
        diff_file.write_text(f"=== REFERENCE ===\n{reference_norm}\n\n=== CURRENT ===\n{output_norm}\n")
        pytest.fail(f"Output differs from reference. See: {diff_file}")


def test_phonopy_irreps_help():
    """Test that help command works."""
    result = subprocess.run(
        [sys.executable, "-m", "anaddb_irreps.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0
    assert "phonopy-irreps" in result.stdout or "usage:" in result.stdout.lower()
