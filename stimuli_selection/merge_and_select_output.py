#!/usr/bin/env python
"""
Script to merge the model outputs and select only those outputs that will be used as stimuli in the eye-tracking
experiment.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Any


def add_to_dict(
        path: str,
        stimuli_dict: dict[str, list[Any]],
        list_attribution: pd.DataFrame,
        selected_items: np.array,
        model: str,
        prompts_introductory_screens_items: pd.DataFrame,
        orig_columns: list[str],
) -> pd.DataFrame:
    assert model in path
    files = os.listdir(path)
    for file in files:
        file_df = pd.read_csv(os.path.join(path, file))
        # subset the model output only to the items selected to be presented in the ET experiment
        file_df = file_df[file_df['item_id'].isin(selected_items)]
        for idx, row in file_df.iterrows():
            item_id = row['item_id']
            list_attr = list_attribution[list_attribution['condition'] == file][item_id].values[0]
            introductory_screen_item = prompts_introductory_screens_items[
                prompts_introductory_screens_items['item_id'] == item_id
            ]['introductory_screen_item'].values[0]
            stimuli_dict['list'].append(list_attr)
            stimuli_dict['model'].append(model)
            stimuli_dict['filename'].append(file)
            stimuli_dict['introductory_screen_item'].append(introductory_screen_item)

            for col in orig_columns:
                stimuli_dict[col].append(row[col])

    return stimuli_dict


def add_questions(
        stimuli_df: pd.DataFrame,
        questions_df: pd.DataFrame,
) -> pd.DataFrame:
    stimuli_df['question'] = ''
    stimuli_df['answer1'] = ''
    stimuli_df['answer2'] = ''
    stimuli_df['answer3'] = ''
    stimuli_df['answer4'] = ''
    stimuli_df['correct_answer'] = ''

    for idx, row in questions_df.iterrows():
        item_id = row['item_id']
        filename = row['filename']
        question = row['question']
        answer1 = row['answer1']
        answer2 = row['answer2']
        answer3 = row['answer3']
        answer4 = row['answer4']
        correct_answer = row['correct_answer']

        stimuli_df.loc[(stimuli_df['item_id'] == item_id) & (stimuli_df['filename'] == filename), 'question'] = question
        stimuli_df.loc[(stimuli_df['item_id'] == item_id) & (stimuli_df['filename'] == filename), 'answer1'] = answer1
        stimuli_df.loc[(stimuli_df['item_id'] == item_id) & (stimuli_df['filename'] == filename), 'answer2'] = answer2
        stimuli_df.loc[(stimuli_df['item_id'] == item_id) & (stimuli_df['filename'] == filename), 'answer3'] = answer3
        stimuli_df.loc[(stimuli_df['item_id'] == item_id) & (stimuli_df['filename'] == filename), 'answer4'] = answer4
        stimuli_df.loc[(stimuli_df['item_id'] == item_id) & (stimuli_df['filename'] == filename), 'correct_answer'] = correct_answer

    return stimuli_df


def main():

    path_to_phi = 'generation/output_generation/phi2'
    path_to_mistral = 'generation/output_generation/mistral'
    path_to_wizardlm = 'generation/output_generation/wizardlm'

    list_attribution = pd.read_csv('stimuli_selection/util_files/list_attribution.csv')
    selected_items = np.load('stimuli_selection/util_files/selected_items.npy')
    prompts_introductory_screens_items = pd.read_csv('stimuli_selection/util_files/prompts_and_introductory_screens.csv')

    all_columns = ['item_id', 'list', 'filename', 'model', 'decoding_strategy', 'introductory_screen_item',
                   'generation_config', 'prompt', 'prompt_toks', 'prompt_input_ids',
                   'gen_seq', 'gen_seq_trunc', 'gen_toks', 'gen_toks_trunc',
                   'gen_toks_trunc_wo_nl', 'gen_toks_trunc_wo_nl_wo_punct', 'gen_ids',
                   'gen_ids_trunc', 'gen_ids_trunc_wo_nl', 'gen_ids_trunc_wo_nl_wo_punct',
                   'tok_idx', 'tok_idx_trunc', 'tok_idx_trunc_wo_nl',
                   'tok_idx_trunc_wo_nl_wo_punct', 'truncated_original',
                   'removed_newlines', 'word_ids_list_wo_nl',
                   'word_ids_list_wo_nl_wo_punct', 'gen_seq_trunc_split_wl',
                   'alignment_mismatch', 'remove_ctr', 'cut_nl_idx', 'type', 'task',
                   'subcategory']
    new_columns = ['model', 'filename', 'introductory_screen_item', 'list']
    orig_columns = [c for c in all_columns if c not in new_columns]

    stimuli_dict = {col: list() for col in all_columns}

    stimuli_dict = add_to_dict(
        path=path_to_phi,
        stimuli_dict=stimuli_dict,
        list_attribution=list_attribution,
        selected_items=selected_items,
        model='phi2',
        prompts_introductory_screens_items=prompts_introductory_screens_items,
        orig_columns=orig_columns,
    )
    stimuli_dict = add_to_dict(
        path=path_to_mistral,
        stimuli_dict=stimuli_dict,
        list_attribution=list_attribution,
        selected_items=selected_items,
        model='mistral',
        prompts_introductory_screens_items=prompts_introductory_screens_items,
        orig_columns=orig_columns,
    )
    stimuli_dict = add_to_dict(
        path=path_to_wizardlm,
        stimuli_dict=stimuli_dict,
        list_attribution=list_attribution,
        selected_items=selected_items,
        model='wizardlm',
        prompts_introductory_screens_items=prompts_introductory_screens_items,
        orig_columns=orig_columns,
    )

    stimuli_df = pd.DataFrame(stimuli_dict)

    # add the comprehension questions to the data frame
    questions_df = pd.read_csv('stimuli_selection/util_files/stimuli_and_questions_balanced.csv')
    stimuli_and_questions_df = add_questions(
        stimuli_df=stimuli_df,
        questions_df=questions_df,
    )

    out_path = 'data/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    stimuli_and_questions_df.to_csv(os.path.join(out_path, 'stimuli.csv'), index=False, sep='\t')


if __name__ == '__main__':
    raise SystemExit(main())
