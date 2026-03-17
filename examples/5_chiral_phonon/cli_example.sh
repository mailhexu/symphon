#!/usr/bin/env zsh
# Example CLI script for chiral phonon analysis with the new Chiral column
# This shows the daughter SG Sohncke class (II-pair or III) in the output

# First, decompress the .xz file if needed
if [ ! -f phonopy_params.yaml ]; then
    echo "Decompressing phonopy_params.yaml.xz..."
    xz -d phonopy_params.yaml.xz 2>/dev/null || echo "File already decompressed or xz not available"
fi

# Run phonopy-irreps with chiral analysis
# The --chiral flag enables the OPD and Daughter SG columns
# The new Chiral column will show:
#   - "II-pair" for Class II Sohncke groups (enantiomorphous pairs)
#   - "III" for Class III Sohncke groups (chiral-supporting)
#   - "-" for non-chiral groups (Class I)
echo "Running phonopy-irreps with chiral analysis..."
#symphon phonopy-irreps --params phonopy_params.yaml --chiral --compare-ground-truth
symphon phonopy-irreps --params phonopy_params.yaml #--chiral --compare-ground-truth
