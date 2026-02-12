#!/usr/bin/env bash
# Simple usage example for the phonopy-irreps CLI.
#
# This assumes you have a phonopy params/YAML file. Here we use the
# BaTiO3_phonopy_params.yaml example bundled with this repository.
#
# Usage:
#   ./phonopy_cli_example.sh
#
# Or run directly:
#   phonopy-irreps --params examples/BaTiO3_phonopy_params.yaml --qpoint 0.0 0.0 0.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARAMS_FILE="${SCRIPT_DIR}/BaTiO3_phonopy_params.yaml"

if [ ! -f "${PARAMS_FILE}" ]; then
  echo "Phonopy params file not found: ${PARAMS_FILE}" >&2
  exit 1
fi

phonopy-irreps --params "${PARAMS_FILE}" --qpoint 0.0 0.0 0.0
