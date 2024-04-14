#!/usr/bin/env python

"""
Script that creates a dictionary for every subject that maps from the Trial_Index_ to the TRIAL_ID, item_id and model.
During the ET experiment, each recorded screen (i.e., both the header and the actual text separately) received a
different trial index. The dictionary serves to double check we map to the correct items when parsing the ascii files.
"""

import pandas as pd
import os
import glob
import pickle

from preprocessing.utils.loading import load_config



def main():

    config = load_config()
    exclude_subjects = config['exclude']['subjects']

    path_to_subjects = 'data/subject_level_data/'
    subj_dirs = os.listdir(path_to_subjects)

    for subj_dir in subj_dirs:

        if subj_dir in exclude_subjects:
            continue

        path_to_subj = os.path.join(path_to_subjects, subj_dir)

        results = pd.read_csv(os.path.join(path_to_subj, 'RESULTS_QUESTIONS.txt'), sep='\t')

        id_to_idx_dict = dict()
        for idx, row in results.iterrows():
            id_to_idx_dict[row['Trial_Index_']] = (row['TRIAL_ID'], row['item_id'], row['model'], row['decoding_strategy'])

        path_save_file = os.path.join(path_to_subj, f'{subj_dir}_idx_to_id.pickle')

        print(f'--- create index to id dict for subject {subj_dir}')

        with open(path_save_file, 'wb') as pickle_file:
            pickle.dump(id_to_idx_dict, pickle_file)


if __name__ == '__main__':
    raise SystemExit(main())
