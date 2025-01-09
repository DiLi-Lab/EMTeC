#!/bin/sh
set -euxo pipefail

python -m preprocessing.parsing.compute_reading_measures
python -m preprocessing.parsing.compute_reading_measures --corrected