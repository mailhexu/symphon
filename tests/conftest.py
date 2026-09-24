"""Pytest configuration for CLI tests."""

import pytest
import symphon._compat  # noqa: F401  (phonopy 4.x alias for spgrep-modulation 0.3.0)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--generate-references",
        action="store_true",
        default=False,
        help="Generate reference output files instead of testing",
    )
