#!/bin/sh
set -euxo pipefail

python -m stimuli_selection.merge_and_select_output
