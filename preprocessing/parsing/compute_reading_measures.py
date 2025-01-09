#!/usr/bin/env python3
"""
Script to compute reading measures from fixation sequences.
"""
from __future__ import annotations

import glob
import os
from argparse import ArgumentParser

import pandas as pd

from preprocessing.utils.loading import load_config


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--check-file-exists',
        action='store_true',
        default=True,
        help='Check whether reading measures for current fixation file have already been computed.',
    )
    parser.add_argument(
        '--corrected',
        action='store_true',
        help='whether or not we are looking at the corrected fixations',
    )
    return parser


def compute_reading_measures(
        fixations_df: pd.DataFrame,
        aoi_df: pd.DataFrame,
        corrected: bool | None = None,
) -> pd.DataFrame:
    """
    Computes reading measures from fixation sequences.
    :param fixations_df: pandas dataframe with columns 'index', 'event_duration', 'word_roi_id', 'word_roi_str'
    :param aoi_df: pandas dataframe with columns 'word_index', 'word', and the aois of each word
    :param corrected: whether or not we are looking at the corrected fixations
    :return: pandas dataframe with reading measures
    """

    # make sure fixations are actually sorted by their index
    fixations_df = fixations_df.sort_values(by=['index'])

    # append one extra dummy fixation to have the next fixation for the actual last fixation
    fixations_df = pd.concat(
        [
            fixations_df,
            pd.DataFrame(
                [[0 for _ in range(len(fixations_df.columns))]], columns=fixations_df.columns,
            ),
        ],
        ignore_index=True,
    )

    # get the original words of the text and their word indices within the text
    text_aois = aoi_df['word_index'].tolist()
    text_strs = aoi_df['word'].tolist()

    # iterate over the words in that text
    word_dict = dict()
    for word_index, word in zip(text_aois, text_strs):
        word_row = {
            'word': word,
            'word_index': word_index,
            'FFD': 0,       # first-fixation duration
            'SFD': 0,       # single-fixation duration
            'FD': 0,        # first duration
            'FPRT': 0,      # first-pass reading time
            'FRT': 0,       # first-reading time
            'TFT': 0,       # total-fixation time
            'RRT': 0,       # re-reading time
            'RPD_inc': 0,   # inclusive regression-path duration
            'RPD_exc': 0,   # exclusive regression-path duration
            'RBRT': 0,      # right-bounded reading time
            'Fix': 0,       # fixation (binary)
            'FPF': 0,       # first-pass fixation (binary)
            'RR': 0,        # re-reading (binary)
            'FPReg': 0,     # first-pass regression (binary)
            'TRC_out': 0,   # total count of outgoing regressions
            'TRC_in': 0,    # total count of incoming regressions
            # 'LP': 0,        # landing position -- cannot have landing position because we don't work with character-based aois
            'SL_in': 0,     # incoming saccade length
            'SL_out': 0,    # outgoing saccade length
            'TFC': 0,       # total fixation count
        }
        
        word_dict[int(word_index)] = word_row

        right_most_word, cur_fix_word_idx, next_fix_word_idx, next_fix_dur = -1, -1, -1, -1

    for index, fixation in fixations_df.iterrows():

        # if aoi is not a number (i.e., it is coded as a missing value using '.'), continue
        try:
            aoi = int(fixation['word_roi_id'])
        except ValueError:
            continue
        # similarly if for the corrected fixations some were not actually mapped to a word, continue (they would result
        # in an error because they are attributed the last interest area + 1)
        if corrected and fixation['word_roi_str'] == '.':
            continue

        # update variables
        last_fix_word_idx = cur_fix_word_idx

        cur_fix_word_idx = next_fix_word_idx
        cur_fix_dur = next_fix_dur

        next_fix_word_idx = aoi
        next_fix_dur = fixation['event_duration']

        # the 0 that we added as dummy fixation at the end of the fixations df
        if next_fix_dur == 0:
            # we set the idx to the idx of the actual last fixation such taht there is no error later
            next_fix_word_idx = cur_fix_word_idx

        if right_most_word < cur_fix_word_idx:
            right_most_word = cur_fix_word_idx

        if cur_fix_word_idx == -1:
            continue

        word_dict[cur_fix_word_idx]['TFT'] += int(cur_fix_dur)

        word_dict[cur_fix_word_idx]['TFC'] += 1

        if word_dict[cur_fix_word_idx]['FD'] == 0:
            word_dict[cur_fix_word_idx]['FD'] += int(cur_fix_dur)

        if right_most_word == cur_fix_word_idx:
            if word_dict[cur_fix_word_idx]['TRC_out'] == 0:
                word_dict[cur_fix_word_idx]['FPRT'] += int(cur_fix_dur)
                if last_fix_word_idx < cur_fix_word_idx:
                    word_dict[cur_fix_word_idx]['FFD'] += int(cur_fix_dur)
        else:
            if right_most_word < cur_fix_word_idx:
                print('error')
            word_dict[right_most_word]['RPD_exc'] += int(cur_fix_dur)

        if cur_fix_word_idx < last_fix_word_idx:
            word_dict[cur_fix_word_idx]['TRC_in'] += 1
        if cur_fix_word_idx > next_fix_word_idx:
            word_dict[cur_fix_word_idx]['TRC_out'] += 1
        if cur_fix_word_idx == right_most_word:
            word_dict[cur_fix_word_idx]['RBRT'] += int(cur_fix_dur)
        if word_dict[cur_fix_word_idx]['FRT'] == 0 and (
            not next_fix_word_idx == cur_fix_word_idx or next_fix_dur == 0
        ):
            word_dict[cur_fix_word_idx]['FRT'] = word_dict[cur_fix_word_idx]['TFT']
        if word_dict[cur_fix_word_idx]['SL_in'] == 0:
            word_dict[cur_fix_word_idx]['SL_in'] = cur_fix_word_idx - last_fix_word_idx
        if word_dict[cur_fix_word_idx]['SL_out'] == 0:
            word_dict[cur_fix_word_idx]['SL_out'] = next_fix_word_idx - cur_fix_word_idx

    # Compute the remaining reading measures from the ones computed above
    for word_indices, word_rm in sorted(word_dict.items()):
        if word_rm['FFD'] == word_rm['FPRT']:
            word_rm['SFD'] = word_rm['FFD']
        word_rm['RRT'] = word_rm['TFT'] - word_rm['FPRT']
        word_rm['FPF'] = int(word_rm['FFD'] > 0)
        word_rm['RR'] = int(word_rm['RRT'] > 0)
        word_rm['FPReg'] = int(word_rm['RPD_exc'] > 0)
        word_rm['Fix'] = int(word_rm['TFT'] > 0)
        word_rm['RPD_inc'] = word_rm['RPD_exc'] + word_rm['RBRT']

        # if it is the first word, we create the df (index of first word is 0)
        if word_indices == 0:
            rm_df = pd.DataFrame([word_rm])
        else:
            rm_df = pd.concat([rm_df, pd.DataFrame([word_rm])])

    return rm_df


def main():

    args = get_parser().parse_args()
    config = load_config()
    subjects_to_exclude = config['exclude']['subjects']

    paths_to_subj_dirs = glob.glob(os.path.join('data', 'subject_level_data', '*'))

    for path_to_subj in paths_to_subj_dirs:
        subj_id = path_to_subj.split('/')[-1]
        if subj_id in subjects_to_exclude:
            continue
        if args.corrected:
            path_to_fix_files = glob.glob(os.path.join(path_to_subj, 'fixations_corrected', 'event_files', '*'))
        else:
            path_to_fix_files = glob.glob(os.path.join(path_to_subj, 'fixations', 'event_files', '*'))
        for fix_file in path_to_fix_files:

            # account for plots folder
            if not fix_file.endswith('csv'):
                continue

            fixations_df = pd.read_csv(fix_file, delimiter='\t')
            TRIAL_ID = fixations_df['TRIAL_ID'].unique().item()
            item_id = fixations_df['item_id'].unique().item()
            Trial_Index_ = fixations_df['Trial_Index_'].unique().item()
            model = fixations_df['model'].unique().item()
            decoding_strategy = fixations_df['decoding_strategy'].unique().item()

            aoi_filename = f'trialid{TRIAL_ID}_{item_id}_trialindex{Trial_Index_}_coordinates.csv'
            aoi_df = pd.read_csv(os.path.join(path_to_subj, 'aoi', aoi_filename), delimiter='\t')

            if args.corrected:
                save_basepath = os.path.join(path_to_subj, 'reading_measures_corrected')
                rm_filename = f'{subj_id}-{item_id}-reading_measures_corrected.csv'
            else:
                save_basepath = os.path.join(path_to_subj, 'reading_measures')
                rm_filename = f'{subj_id}-{item_id}-reading_measures.csv'
            path_save_rm_file = os.path.join(save_basepath, rm_filename)

            if args.check_file_exists:
                if os.path.isfile(path_save_rm_file):
                    print(f'\t--- file {path_save_rm_file} already exists. skipping.')
                    continue

            if not os.path.exists(save_basepath):
                os.makedirs(save_basepath)

            print(f'---processing file {path_save_rm_file}')

            if args.corrected:
                corrected = True
            else:
                corrected = False

            rm_df = compute_reading_measures(
                fixations_df=fixations_df,
                aoi_df=aoi_df,
                corrected=corrected,
            )

            rm_df['subject_id'] = subj_id
            rm_df['item_id'] = item_id
            rm_df['TRIAL_ID'] = TRIAL_ID
            rm_df['Trial_Index_'] = Trial_Index_
            rm_df['model'] = model
            rm_df['decoding_strategy'] = decoding_strategy

            rm_df.to_csv(path_save_rm_file, index=False)


if __name__ == '__main__':
    raise SystemExit(main())
