#!/bin/bash
# Step 1: create the Python 3.9 venv for training and evaluation ($TG_ENV), with the exact package
# versions the results were produced with (requirements/tower-env.txt). Run on a login node (~10 min).
# If $TG_ENV already exists it is left untouched and only compared against the pinned versions.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./config.sh

if [ -x "$TG_ENV/bin/python" ]; then
    echo "$TG_ENV already exists -- not modifying it. Differences from requirements/tower-env.txt:"
    DIFF=$(diff <(sort -f requirements/tower-env.txt) <("$TG_ENV/bin/pip" freeze 2>/dev/null | sort -f) \
        | grep '^[<>]' | sed 's/^</  expected: /; s/^>/  installed:/' || true)
    echo "${DIFF:-  none}"
    exit 0
fi

PY=/usr/bin/python3  # Python 3.9 on Snellius (RHEL 9)
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 9), sys.version' \
    || { echo "expected Python 3.9 at $PY; set PY in this script to a Python 3.9 interpreter"; exit 1; }
"$PY" -m venv "$TG_ENV"
"$TG_ENV/bin/pip" install -q -U pip
"$TG_ENV/bin/pip" install -r requirements/tower-env.txt
echo "done: $TG_ENV"
