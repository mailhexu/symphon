#!/usr/bin/env bash
# Simple usage example for the anaddb-irreps CLI.
#
# This assumes you have already run anaddb to generate a PHBST file.
# Here we use the MoS2_1T example bundled with this repository.
#
# Usage:
#   ./cli_example.sh
#
# Or run directly:
#   anaddb-irreps -p examples/MoS2_1T/run_PHBST.nc -q 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHBST_FILE="${SCRIPT_DIR}/MoS2_1T/run_PHBST.nc"

if [ ! -f "${PHBST_FILE}" ]; then
  echo "PHBST file not found: ${PHBST_FILE}" >&2
  echo "Make sure you generated run_PHBST.nc using anaddb before running this script." >&2
  exit 1
fi

anaddb-irreps -p "${PHBST_FILE}" -q 0
