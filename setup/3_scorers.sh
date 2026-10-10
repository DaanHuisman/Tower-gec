#!/bin/bash
# Step 3: install the official GEC scorers into $TG_SCORERS, in their own Python 3.12 venv
# (ChERRANT pins packages that clash with the training env). Run on a login node (~10 min).
#
#   ChERRANT (Chinese, char-level P/R/F0.5)   github.com/HillZhang1999/MuCGEC            @ 3922193
#     + patches/cherrant_char_level_without_ltp.patch: moves `from ltp import LTP` into the word-level
#       branch. Char-level scoring never uses LTP, so scores are unchanged; this avoids installing ltp.
#   MultiGEC-2025 GLEU + ERRANT (Italian)     github.com/spraakbanken/multigec-2025      @ a9aa8b5
#     + patches/multigec_lowercase_filename_only.patch: lowercases only the hypothesis .tmp filename, not
#       its whole path, so scoring works when the repo path has capitals. Scores are unchanged.
#     + its ERRANT / spacy_conll forks        github.com/cainesap/{errant,spacy_conll}   @ 16d835e / bd3e49f
#     + UDPipe Italian model
# If $TG_SCORERS/venv already exists, nothing is changed.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./config.sh

if [ -x "$TG_SCORERS/venv/bin/python" ]; then
    echo "[skip] $TG_SCORERS/venv already exists -- not modifying it"
    exit 0
fi
mkdir -p "$TG_SCORERS"

clone_at() {  # clone_at <url> <dir> <commit>
    [ -d "$2/.git" ] && { echo "[skip] $2 exists"; return; }
    git init -q "$2"
    git -C "$2" remote add origin "$1"
    git -C "$2" fetch -q --depth 1 origin "$3"
    git -C "$2" checkout -q FETCH_HEAD
}

clone_at https://github.com/HillZhang1999/MuCGEC.git "$TG_SCORERS/MuCGEC" 39221934ee4ccf07801879136a75dd7e11319977
PATCH="$TG_ROOT/patches/cherrant_char_level_without_ltp.patch"
if git -C "$TG_SCORERS/MuCGEC" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "[skip] ChERRANT already patched"
else
    git -C "$TG_SCORERS/MuCGEC" apply "$PATCH"
fi

clone_at https://github.com/spraakbanken/multigec-2025.git "$TG_SCORERS/multigec-2025" a9aa8b5a9eef95e1d47e305e838a5837096d6340
PATCH="$TG_ROOT/patches/multigec_lowercase_filename_only.patch"
if git -C "$TG_SCORERS/multigec-2025" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "[skip] MultiGEC already patched"
else
    git -C "$TG_SCORERS/multigec-2025" apply "$PATCH"
fi
G="$TG_SCORERS/multigec-2025/scripts/local_eval/gleu_errant"
clone_at https://github.com/cainesap/errant "$G/errant" 16d835e471d729b18d474aa085b1f6cc89e3fab0
clone_at https://github.com/cainesap/spacy_conll "$G/spacy_conll" bd3e49f52684e7f0b3347b98b1f466083d665645
mkdir -p "$G/spacy_udpipe_models"  # the MultiGEC ERRANT fork loads UDPipe models relative to this dir

# Python 3.12 venv (the venv keeps working after the module is unloaded)
module purge
module load $TG_PY312_MODULES
python3 -m venv "$TG_SCORERS/venv"
module purge
V="$TG_SCORERS/venv/bin"
"$V/pip" install -q -U pip
"$V/pip" install -r requirements/scorers.txt
"$V/pip" install -q --no-deps -e "$G/errant" -e "$G/spacy_conll"
"$V/python" -c "import spacy_udpipe; spacy_udpipe.download('it')"
echo "done: $TG_SCORERS"
