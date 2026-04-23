#!/usr/bin/env zsh
# Example CLI script for circular phonon analysis
# Two modes: abstract (group theory) and concrete (from PHBST data)

# --- Abstract: Group-theory prediction (no data files needed) ---

# Find circular phonon candidates for a specific space group
echo "=== Abstract: Circular phonon candidates for SG 198 (P2_1_3) ==="
symphon circular-abstract --sg 198

echo ""
echo "=== Abstract: Circular phonon candidates for SG 19 (P2_12_12_1) ==="
symphon circular-abstract --sg 19

# Scan all 65 Sohncke (chiral-supporting) groups
echo ""
echo "=== All 65 Sohncke groups with circular phonon candidates ==="
symphon circular-abstract --all-sohncke

# --- Concrete: Full analysis from PHBST file ---
# Requires an ABINIT PHBST NetCDF file:
#
#   symphon-circular run_PHBST.nc
#   symphon-circular run_PHBST.nc --qpoint 0.5 0.0 0.0
#   symphon-circular run_PHBST.nc --symprec 1e-8 --verbose
