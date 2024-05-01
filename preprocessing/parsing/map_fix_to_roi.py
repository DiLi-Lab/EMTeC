#!/usr/bin/env python3
from __future__ import annotations

import os

import pandas as pd


# for decoding
def get_word_rois(
        df_subset: pd.DataFrame,
        aoi_df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Assigns a word ROI to each fixation position.
    :param df_subset: pandas dataframe with columns 'fix_mean_x', 'fix_mean_y'
    :param aoi_dict: dictionary with word ROI coordinates
    :return: list of word ROI ids, list of word ROI strings
    """
    xs = df_subset.fix_mean_x.to_numpy(dtype=float)
    ys = df_subset.fix_mean_y.to_numpy(dtype=float)
    fix_rois: list = ['.'] * len(xs)
    fix_rois_str: list = ['.'] * len(xs)
    assert len(xs) == len(ys)

    # for each fixation position
    for fix_index, (current_fix_x, current_fix_y) in enumerate(zip(xs, ys)):
        found_roi = False

        for row_idx, row in aoi_df.iterrows():
            y_top = row['y_top']
            y_bottom = row['y_bottom']
            x_left = row['x_left']
            x_right = row['x_right']
            if (
                    ((current_fix_x - x_left) >= 0)
                    & ((current_fix_x - x_right) <= 0)
                    & ((current_fix_y - y_top) >= 0)
                    & ((current_fix_y - y_bottom) <= 0)
            ):
                fix_rois[fix_index] = row['word_index']
                fix_rois_str[fix_index] = row['word']
                found_roi = True
                break
        if found_roi:
            continue

    return fix_rois, fix_rois_str


# for decoding
def load_aois(
        subj_dir: str,
        TRIAL_ID: int,
        item_id: str,
        Trial_Index_: int,
) -> pd.DataFrame:
    """
    Load the file with the word-level AOIs for one trial.
    :param subj_dir: path to the subject directory
    :param TRIAL_ID: trial id
    :param item_id: item id
    :param Trial_Index_: trial index
    :return: a dataframe with the word-level AOIs
    """
    filename = f'trialid{TRIAL_ID}_{item_id}_trialindex{Trial_Index_}_coordinates.csv'
    path_to_aois = os.path.join(subj_dir, 'aoi', filename)
    aoi_df = pd.read_csv(path_to_aois, delimiter='\t')
    return aoi_df


def get_aois_from_event_data(
        event_dat: pd.DataFrame,
        subj_dir: str,
        TRIAL_ID: int,
        item_id: str,
        Trial_Index_: int,
) -> pd.DataFrame:
    """
    Assigns a word ROI to each fixation position.
    :param event_dat: pandas dataframe with columns 'fix_mean_x', 'fix_mean_y'
    :param subj_dir: path to the subject directory
    :param TRIAL_ID: trial id
    :param item_id: item id
    :param Trial_Index_: trial index
    :return: pandas dataframe with columns 'word_roi_id', 'word_roi_str'
    """
    word_roi_df = load_aois(
        subj_dir=subj_dir,
        TRIAL_ID=TRIAL_ID,
        item_id=item_id,
        Trial_Index_=Trial_Index_,
    )
    word_roi_ids, word_roi_str = get_word_rois(
        df_subset=event_dat,
        aoi_df=word_roi_df,
    )
    event_dat['word_roi_id'] = word_roi_ids
    event_dat['word_roi_str'] = word_roi_str
    return event_dat
