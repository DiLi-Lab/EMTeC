#!/bin/sh
set -euxo pipefail

# merge participant information
python -m preprocessing.utils.merge_participant_info

# merge the subject-trial-level fixation sequence files into one big file. Merge both corrected and uncorrected fixations.
python -m preprocessing.utils.merge_fixation_files
python -m preprocessing.utils.merge_fixation_files --corrected

# merge the subject-trial-level reading measure files into one big file. Merge the reading measures computed from both
# the corrected and uncorrected fixations
python -m preprocessing.utils.merge_rm_files
python -m preprocessing.utils.merge_rm_files --corrected

# merge the reading measures file with the word-level annotations (frequencies, surprisal values, ...) and the
# prompt type information
python -m preprocessing.utils.merge_annotations
# remove the word-level annotation file
rm annotation/word_level_annotations.csv
