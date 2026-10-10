#!/bin/bash
# Step 3b: official ERRANT for the Tower paper's GEC test sets (English CoNLL-2014, German, Spanish).
# Run on a login node (needs internet).
#
#   errant 3.0.0 + spaCy 3.7 + en/de/es small models   -> $TG_SCORERS/errant-venv   (~1 min, 0.6 GB)
#
# This is a separate venv on purpose: $TG_SCORERS/venv has the MultiGEC fork of ERRANT, which the
# Italian scores depend on. tower-eval, which produced the paper's numbers, requires errant>=2.3.3 and
# spacy^3.7.2. If $TG_SCORERS/errant-venv already exists, nothing is changed.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./config.sh

V="$TG_SCORERS/errant-venv"
if [ -x "$V/bin/errant_compare" ]; then
    echo "[skip] $V already exists -- not modifying it"
    exit 0
fi
mkdir -p "$TG_SCORERS"
module purge
module load $TG_PY312_MODULES
python3 -m venv "$V"
module purge
"$V/bin/pip" install -q -U pip
"$V/bin/pip" install -q -r requirements/errant.txt
echo "done: $V"
