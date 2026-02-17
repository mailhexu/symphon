"""Pytest configuration for CLI tests."""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--generate-references",
        action="store_true",
        default=False,
        help="Generate reference output files instead of testing",
    )
