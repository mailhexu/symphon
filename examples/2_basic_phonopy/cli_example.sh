#!/usr/bin/env bash
# Simple usage example for the phonopy-irreps CLI.
#
# This demonstrates the new automatic high-symmetry point discovery feature.
# The tool will automatically:
#   - Discover all high-symmetry k-points (GM, M, R, X, etc.)
#   - Show dual labels (Mulliken + BCS) at Gamma point
#   - Show BCS labels at other high-symmetry points
#
# Usage:
#   ./cli_example.sh
#
# Or run directly:
#   phonopy-irreps --params BaTiO3_phonopy_params.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARAMS_FILE="${SCRIPT_DIR}/BaTiO3_phonopy_params.yaml"

if [ ! -f "${PARAMS_FILE}" ]; then
  echo "Phonopy params file not found: ${PARAMS_FILE}" >&2
  exit 1
fi

phonopy-irreps --params "${PARAMS_FILE}"
