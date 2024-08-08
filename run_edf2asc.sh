#!/bin/sh
set -euxo pipefail

python -m preprocessing.utils.convert_edf_files_to_asc
