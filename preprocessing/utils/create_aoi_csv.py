#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script that creates CSVs containing the areas of interest for each trial on word level as opposed to the original
character level. Needed to map fixations to the word they landed on.
"""


import pandas as pd
import os
import glob
import pickle
from argparse import ArgumentParser


def main():

    # read in the stimuli
    stimuli = pd.read_csv('stimuli_selection/files/stimuli_and_questions.tsv', sep='\t')

    paths_to_subjects = glob.glob(os.path.join('data', 'subject_level_data', '*'))

    for subject in paths_to_subjects:
        results = pd.read_csv(os.path.join(subject, 'RESULTS_QUESTIONS.txt'), sep='\t')
        path_to_ias_files = os.path.join(subject, 'aoi')

        for trial_id in results['TRIAL_ID'].tolist():

            item_id = results[results['TRIAL_ID'] == trial_id]['item_id'].item()
            filename = results[results['TRIAL_ID'] == trial_id]['filename'].item()
            trial_index = results[results['TRIAL_ID'] == trial_id]['Trial_Index_'].item()

            aoi_df_filename = f'trialid{trial_id}_{item_id}_trialindex{trial_index}_coordinates.csv'

            if os.path.isfile(os.path.join(path_to_ias_files, aoi_df_filename)):
                print(f'---file {aoi_df_filename} already exists. skipping.')
                continue

            print(f'--- processing subject {subject}, trial id {trial_id}')

            gen_seq_trunc_str = stimuli[(stimuli['filename'] == filename) & (stimuli['item_id'] == item_id)][
                'gen_seq_trunc'].tolist()[0]

            ias_filename = f'IA_{trial_index}.ias'
            with open(os.path.join(path_to_ias_files, ias_filename)) as f:
                ias_char_list = [line.rstrip().split('\t') for line in f.readlines()]

            aoi_dict = {
                'word_index': list(),
                'y_top': list(),
                'y_bottom': list(),
                'x_left': list(),
                'x_right': list(),
                'word': list(),
            }

            x_left, x_right, y_top, y_bottom = 0, 0, 0, 0
            start_idx, end_idx = 0, 0
            for word_idx, word in enumerate(gen_seq_trunc_str.split()):
                if word_idx == 0:
                    start_idx = 0
                    end_idx = len(word) - 1
                else:
                    start_idx = end_idx + 1
                    end_idx = start_idx + len(word) - 1
                y_top = ias_char_list[start_idx][3]
                y_bottom = ias_char_list[start_idx][5]
                x_left = ias_char_list[start_idx][2]
                x_right = ias_char_list[end_idx][4]

                aoi_dict['word_index'].append(word_idx)
                aoi_dict['y_top'].append(y_top)
                aoi_dict['y_bottom'].append(y_bottom)
                aoi_dict['x_left'].append(x_left)
                aoi_dict['x_right'].append(x_right)
                aoi_dict['word'].append(word)

            aoi_df = pd.DataFrame.from_dict(aoi_dict)

            aoi_df.to_csv(os.path.join(path_to_ias_files, aoi_df_filename), index=False, sep='\t')


if __name__ == '__main__':
    raise SystemExit(main())
