#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge the subject-trial-level reading measure files into one big file.
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

    rms_dfs = list()


    path_to_subjects = glob.glob(os.path.join('data', 'subject_level_data', '*'))
    for path_to_subject in path_to_subjects:
        subj_id = path_to_subject.split('/')[-1]
        if subj_id in exclude_subjects:
            continue
        if args.corrected:
            paths_to_rms = glob.glob(os.path.join(path_to_subject, 'reading_measures_corrected', '*'))
        else:
            paths_to_rms = glob.glob(os.path.join(path_to_subject, 'reading_measures', '*'))
        for path_to_rms in paths_to_rms:
            print(f'--- merging {path_to_rms}')
            rm_df = pd.read_csv(path_to_rms)
            rms_dfs.append(rm_df)

    all_rms = pd.concat(rms_dfs)

    # rename and drop columns
    all_rms = all_rms.rename(columns={'word_index': 'word_id'})
    all_rms = all_rms.drop(columns=['Trial_Index_'])

    # change order of columns
    columns = [
        'subject_id', 'item_id', 'model', 'decoding_strategy', 'TRIAL_ID', 'word_id', 'word', 'FFD', 'SFD',
        'FD', 'FPRT', 'FRT', 'TFT', 'RRT', 'RPD_inc', 'RPD_exc', 'RBRT', 'Fix', 'FPF', 'RR', 'FPReg', 'TRC_out',
        'TRC_in', 'SL_in', 'SL_out', 'TFC'
       ]
    all_rms = all_rms[columns]

    if args.corrected:
        save_path = os.path.join('data', 'reading_measures_corrected.csv')
    else:
        save_path = os.path.join('data', 'reading_measures.csv')
    all_rms.to_csv(save_path, sep='\t', index=False)


if __name__ == '__main__':
    raise SystemExit(main())
