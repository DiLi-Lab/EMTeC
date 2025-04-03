# test comment
#!/bin/sh
set -euxo pipefail

python3 -m preprocessing.utils.create_trialid_to_index_dict
python3 -m preprocessing.parsing.preprocess_asc_files > asc_error.log 2>&1
