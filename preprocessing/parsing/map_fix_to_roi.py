#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from typing import Dict, Collection, List, Tuple
from os.path import exists
import pickle


def get_extensions(df, row, first_line, last_line, extension_size):
    """
    In data viewer, AOIs in first and last line were skewed to top/bottom (by 9 pixels each). If you need to adjust
    the AOIs further, make sure you make everything symmetric. Here, our AOIs are 35 pixels in length.
    return: (extension at top, extension at bottom)
    """
    extend = extension_size
    if df.iloc[row][3] == first_line:
        # if word is on the first line, extend top minus skew
        return extend-9, extend  # TODO REMOVE HARDCODING
    elif df.iloc[row][3] == last_line:
        # if word is on the first line, don't extend top
        return extend, extend-9
    else:
        # symmetric extension if not first or last line
        return extend, extend


def get_word_aoi_dict() -> Dict[int, Dict[int, Tuple[np.array, List[str]]]]:
    word_XY_dict: Dict[int, Dict[int, Tuple[np.array, List[str]]]] = dict()
    for text in range(1, 17):
        word_XY_dict[text] = dict()
        for screenid in range(1, 6):
            roifile = f"../aoi/text_{text}_{screenid}.ias"
            adf = pd.read_csv(
                roifile,
                delimiter="\t",
                engine="python",
                encoding="utf-8",
                # encoding="latin1",
                header=None,
            )
            # get top of first/last line
            first_line_top = adf[3].min()
            last_line_top = adf[3].max()
            word_XY_l = []
            words_l = []
            characters_in_word = 0
            word = ""
            # 1 row = 1 character
            for row in range(len(adf)):
                if adf.iloc[row][6] != "_":
                    # if not end of word
                    characters_in_word += 1
                    word += adf.iloc[row][6]
                    cur_x_right = adf.iloc[row][4]
                    cur_y_top = adf.iloc[row][3]
                    cur_y_bot = adf.iloc[row][5]
                else:
                    # if end of word
                    # add extension to top and bottom of line
                    extend_top, extend_bottom = get_extensions(adf, row, first_line_top, last_line_top, 16)
                    cur_x_left = adf.iloc[row - characters_in_word][2]
                    word_XY_l.append(
                        [[cur_x_left, cur_x_right], [cur_y_top-extend_top, cur_y_bot+extend_bottom]],
                    )
                    words_l.append(word)
                    word = ""
                    characters_in_word = 0
            # after last iteration, store last word (no "-" in text)
            cur_x_left = adf.iloc[len(adf) - characters_in_word][2]
            extend_top, extend_bottom = get_extensions(adf, len(adf)-1, first_line_top, last_line_top, 16)
            # assert extend_bottom == 0, "No extension on last line allowed"
            word_XY_l.append(
                [[cur_x_left, cur_x_right], [cur_y_top-extend_top, cur_y_bot+extend_bottom]],
            )
            words_l.append(word)
            word_XY = np.array(word_XY_l)
            word_XY_dict[text][screenid] = (word_XY, words_l)
    return word_XY_dict  # dimensions: [textid][screenid][[x_left x_right][y_top y_bottom]]


# for indico
# def get_word_rois(df_subset, word_rois, word_rois_str) -> Tuple[List[str], List[str]]:
#     xs = df_subset.fix_mean_x.to_numpy(dtype=float)
#     ys = df_subset.fix_mean_y.to_numpy(dtype=float)
#     fix_rois: List = ["."] * len(xs)
#     fix_rois_str: List = ["."] * len(xs)
#     assert len(xs) == len(ys)
#     # for each fixation
#     for i in range(0, len(xs)):
#         found_roi = False
#         current_fix_x = xs[i]
#         current_fix_y = ys[i]
#         # loop through roi dict to find associated roi
#         for j in range(len(word_rois)):
#             if (
#                 ((current_fix_x - word_rois[j][0][0]) >= 0)
#                 & ((current_fix_x - word_rois[j][0][1]) <= 0)
#                 & ((current_fix_y - word_rois[j][1][0]) >= 0)
#                 & ((current_fix_y - word_rois[j][1][1]) <= 0)
#             ):
#                 fix_rois[i] = str(j+1)
#                 fix_rois_str[i] = word_rois_str[j]
#                 # print(f'found roi {j} for fixation {i}')
#                 found_roi=True
#                 break
#         if found_roi:
#             continue
#     return fix_rois, fix_rois_str

# for decoding
def get_word_rois(
        df_subset: pd.DataFrame,
        aoi_df: pd.DataFrame,
        #aoi_dict: Dict[Tuple[int], Tuple[str]],
):
    xs = df_subset.fix_mean_x.to_numpy(dtype=float)
    ys = df_subset.fix_mean_y.to_numpy(dtype=float)
    fix_rois: List = ["."] * len(xs)
    fix_rois_str: List = ["."] * len(xs)
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
        #
        #
        # # iterate through all keys in the aoi dict
        # # each key is  auple (y_top, y_bottom, x_left, x_right); each value is (word_idx, word)
        # for key in aoi_dict.keys():
        #     y_top = int(key[0])
        #     y_bottom = int(key[1])
        #     x_left = int(key[2])
        #     x_right = int(key[3])
        #     if (
        #             ((current_fix_x - x_left) >= 0)
        #             & ((current_fix_x - x_right) <= 0)
        #             & ((current_fix_y - y_top) >= 0)
        #             & ((current_fix_y - y_bottom) <= 0)
        #     ):
        #         fix_rois[fix_index] = aoi_dict[key][0]
        #         fix_rois_str[fix_index] = aoi_dict[key][1]
        #         found_roi = True
        #         break
        # if found_roi:
        #     continue
    return fix_rois, fix_rois_str



def load_word_aois()-> Dict[int, Dict[int, Tuple[np.array, List[str]]]]:
    word_XY_dict: Dict[int, Dict[int, Tuple[np.array, List[str]]]] = dict()
    for text in range(1, 17):
        word_XY_dict[text] = dict()
        for screenid in range(1, 6):
            words_l = []
            word_XY = []
            roifile = f"./aoi/text_{text}_{screenid}.ias"
            adf = pd.read_csv(
                roifile,
                delimiter="\t",
                engine="python",
                encoding="utf-8",
                # encoding="latin1",
                header=None,
            )
            for row in range(len(adf)):
                # word_id = adf.iloc[row][1]
                words_l.append(adf.iloc[row][6])
                cur_x_left = adf.iloc[row][2]
                cur_x_right = adf.iloc[row][4]
                cur_y_top = adf.iloc[row][3]
                cur_y_bot = adf.iloc[row][5]
                word_XY.append([[cur_x_left, cur_x_right], [cur_y_top, cur_y_bot]])
            word_XY = np.array(word_XY)
            word_XY_dict[text][screenid] = (word_XY, words_l)
    return word_XY_dict  # dimensions: [textid][screenid][[x_left x_right][y_top y_bottom]]


# for indico
# def load_aois():
#     if exists("./aoi/xy_dict.pickle"):
#         with open("./aoi/xy_dict.pickle", "rb") as handle:
#             word_XY_dict = pickle.load(handle)
#             return word_XY_dict
#
#     else:
#         # word_XY_dict = get_word_aoi_dict()
#         word_XY_dict = load_word_aois()
#         with open("./aoi/xy_dict.pickle", "wb") as handle:
#             pickle.dump(word_XY_dict, handle)
#             return word_XY_dict


# for decoding
def load_aois(
        subj_dir: str,
        TRIAL_ID: int,
        item_id: str,
        Trial_Index_: int,
):
    #filename = f'trialid{TRIAL_ID}_{item_id}_trialindex{Trial_Index_}_coordinates.pickle'
    filename = f'trialid{TRIAL_ID}_{item_id}_trialindex{Trial_Index_}_coordinates.csv'
    path_to_aois = os.path.join(subj_dir, 'aoi', filename)
    # with open(path_to_aois, 'rb') as f:
    #     aoi_dict = pickle.load(f)
    aoi_df = pd.read_csv(path_to_aois, delimiter='\t')
    return aoi_df
    #return aoi_dict


# def get_aois_from_event_data(event_dat: pd.DataFrame, text_id: int, screen_id: int) -> pd.DataFrame:
#     word_roi_dict = load_aois()
#     word_rois = word_roi_dict[text_id][screen_id][0]
#     word_rois_str = word_roi_dict[text_id][screen_id][1]
#     word_roi_ids, word_rois_str = get_word_rois(event_dat, word_rois, word_rois_str)
#     event_dat['word_roi_id'] = word_roi_ids
#     event_dat['word_roi_str'] = word_rois_str
#     return event_dat


def get_aois_from_event_data(
        event_dat: pd.DataFrame,
        subj_dir: str,
        TRIAL_ID: int,
        item_id: str,
        Trial_Index_: int,
):
    # word_roi_dict = load_aois(
    #     subj_dir=subj_dir,
    #     TRIAL_ID=TRIAL_ID,
    #     item_id=item_id,
    #     Trial_Index_=Trial_Index_,
    # )
    word_roi_df = load_aois(
        subj_dir=subj_dir,
        TRIAL_ID=TRIAL_ID,
        item_id=item_id,
        Trial_Index_=Trial_Index_,
    )
    # word_roi_ids, word_roi_str = get_word_rois(
    #     df_subset=event_dat,
    #     aoi_dict=word_roi_dict,
    # )
    word_roi_ids, word_roi_str = get_word_rois(
        df_subset=event_dat,
        aoi_df=word_roi_df,
    )
    event_dat['word_roi_id'] = word_roi_ids
    event_dat['word_roi_str'] = word_roi_str
    return event_dat

