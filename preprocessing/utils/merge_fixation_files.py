#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge the subject-trial-level fixation sequence files into one big file.
"""


import os
import glob
from argparse import ArgumentParser
import pandas as pd
import numpy as np

from preprocessing.utils.loading import load_config


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--corrected',
        action='store_true',
        help='whether or not we are looking at the corrected fixations'
    )
    return parser


def main():

    args = get_parser().parse_args()
    config = load_config()
    exclude_subjects = config['exclude']['subjects']

    fixations_dfs = list()


    path_to_subjects = glob.glob(os.path.join('data', 'subject_level_data', '*'))
    for path_to_subject in path_to_subjects:
        subj_id = path_to_subject.split('/')[-1]
        if subj_id in exclude_subjects:
            continue

        if args.corrected:
            paths_to_fixs = glob.glob(os.path.join(path_to_subject, 'fixations_corrected', 'event_files', '*'))
        else:
            paths_to_fixs = glob.glob(os.path.join(path_to_subject, 'fixations', 'event_files', '*'))

        for path_to_fix in paths_to_fixs:
            print(f'--- merging {path_to_fix}')
            fix_df = pd.read_csv(path_to_fix, delimiter='\t')
            fixations_dfs.append(fix_df)

    all_fixs = pd.concat(fixations_dfs)
    breakpoint()

    # rename certain columns
    all_fixs = all_fixs.rename(
        columns={
            'index': 'fixation_index',
            'event_duration': 'fixation_duration',
            'word_roi_id': 'word_id',
            'word_roi_str': 'word',
        }
    )

    # remove unnecessary columns
    all_fixs = all_fixs.drop(columns=['Trial_Index_', 'event', 'event_len', 'fix_mean_x', 'fix_mean_y', 't_end', 't_start',])


    # change the order of columns
    columns = ['subject_id', 'item_id', 'model', 'decoding_strategy', 'TRIAL_ID', 'fixation_index', 'fixation_duration', 'word_id', 'word']
    all_fixs = all_fixs[columns]

    if args.corrected:
        save_path = os.path.join('data', 'fixations_corrected.csv')
    else:
        save_path = os.path.join('data', 'fixations.csv')
    all_fixs.to_csv(save_path, sep='\t', index=False)


if __name__ == '__main__':
    raise SystemExit(main())
