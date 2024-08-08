#!/bin/sh
set -euxo pipefail

python -m preprocessing.fixation_correction.fixation_correction \
--run-on-subj ET_01
