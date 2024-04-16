#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge the participant results to the questions and their questionnaire information.
"""

import os
import glob
import pandas as pd
from preprocessing.utils.loading import load_config


def main():

    config = load_config()

    exclude_subjects = config['exclude']['subjects']
    exclude_screens = config['exclude']['screens']

    questions_dfs = list()
    demographics_dfs = list()

    path_to_subjects = glob.glob(os.path.join('data', 'subject_level_data', '*'))
    for path_to_subject in path_to_subjects:
        subj_id = path_to_subject.split('/')[-1]
        if subj_id in exclude_subjects:
            continue
        questions_df = pd.read_csv(os.path.join(path_to_subject, 'RESULTS_QUESTIONS.txt'), delimiter='\t')
        demographics_df = pd.read_csv(os.path.join(path_to_subject, 'RESULTS_QUESTIONNAIRE.txt'), delimiter='\t')

        questions_dfs.append(questions_df)
        demographics_dfs.append(demographics_df)

    all_questions = pd.concat(questions_dfs)
    all_demographics = pd.concat(demographics_dfs)

    all_questions = all_questions.drop(columns=['SUBJ_ID', 'Trial_Recycled_', 'prompt'])
    all_questions = all_questions.rename(columns={'Session_Name_': 'subject_id'})

    all_demographics = all_demographics.drop(columns=['item_id', 'filename', 'model', 'decoding_strategy',
                                                      'introductory_screen_item', 'prompt', 'gen_seq_trunc', 'question',
                                                      'answer1', 'answer2', 'answer3', 'answer4', 'correct_answer'])
    all_demographics = all_demographics.rename(columns={'Session_Name_': 'subject_id'})

    # exclude screens
    for subj_to_exclude in exclude_screens.keys():
        for screen_to_exclude in exclude_screens[subj_to_exclude]:
            all_questions = all_questions[~((all_questions['subject_id'] == subj_to_exclude) & (all_questions['TRIAL_ID'] == screen_to_exclude))]


    out_path = os.path.join('data', 'participant_info')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    all_questions.to_csv(os.path.join(out_path, 'participant_results.csv'), sep='\t', index=False)
    all_demographics.to_csv(os.path.join(out_path, 'participant_info.csv'), sep='\t', index=False)


if __name__ == '__main__':
    raise SystemExit(main())
