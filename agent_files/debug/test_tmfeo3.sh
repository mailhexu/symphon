#!/bin/bash
# Test script for TmFeO3 phonopy irrep analysis
#
# Tests auto-discovery of high-symmetry k-points for TmFeO3 (Pnma, space group 62)
# with dual labels at Gamma point.
#
# How to run:
#   bash agent_files/debug/test_tmfeo3.sh
#
# What it tests:
# - Auto-discovery of high-symmetry k-points
# - Dual labels (Mulliken + BCS) at Gamma
# - Clean output (no redundant qx/qy/qz columns)
# - Uses symprec=1e-2 for proper space group detection

echo "======================================================================"
echo "Testing phonopy-irreps with TmFeO3 (Pnma, space group 62)"
echo "======================================================================"
echo ""

uv run phonopy-irreps \
  --params examples/3_advanced_phonopy/TmFeO3_phonopy_params.yaml \
  --symprec 1e-2

echo ""
echo "======================================================================"
echo "Test completed!"
echo "======================================================================"
