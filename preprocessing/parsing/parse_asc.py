#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eyetracking-Data extraction

This script extracts the lines with samples (timestamp, x-screen coordinate, y-screen coordinate, pupil diameter)
from Eyelink Portable Duo raw data files (previously converted from edf to ascii).

Extract recording samples on the texts;
remove samples on practice trials and on headers;
remove samples between trials (calibration)
"""

import re
import os
import codecs
import pickle
import logging
from collections import defaultdict
from typing import Dict, List, Union

import pandas as pd
import numpy as np

import preprocessing.parsing.message_patterns as mp


def parse_asc_file(
        filepath: str,
        experiments: List[str],
        columns: Union[str, Dict[str, str]],
        exclude_screens: Dict[str, List[int]],
        check_file_exists: bool = True,
):
    filename = filepath.split('/')[-1]
    subject_id = filename[:-4]
    filepath_csv = filepath[:-4] + '.csv'
    logging.basicConfig(format='%(levelname)s::%(message)s',
                        level=logging.INFO)
    logging.info(f'parsing file {filename}')

    # Skip if all csv files exist
    csv_file_exists = []
    if check_file_exists:
        csv_file_exists.append(os.path.isfile(filepath_csv))

    if all(csv_file_exists):
        logging.info(f'all csv files for {filename} exist. skipping.')
        return 0

    print(filepath)

    item_id = -1
    TRIAL_ID = -1
    Trial_Index_ = -1
    trial_ctr = 0

    # STORING
    data_out = defaultdict(lambda: defaultdict(list))
    select = False  # true if next eyetracking sample should be written out
    curr_exp = ''  # current experiment to write sample for

    asc_file = codecs.open(filepath, 'r', encoding='ascii', errors='ignore')

    # load dictionary that maps Trial_Index_ to TRIAL_ID and item_id as sanity check
    path_to_mapping_dict = filepath[:-4] + '_idx_to_id.pickle'
    with open(path_to_mapping_dict, 'rb') as pickle_file:
        idx_to_id_dict = pickle.load(pickle_file)

    line = True
    while line:
        try:
            line = asc_file.readline()
        except UnicodeDecodeError:
            logging.error(f'DECODING ERROR, aborting file {filename}')
            return -1

        # if the line is a message
        if line.startswith('MSG'):

            if mp.on_reading_text.match(line):
                curr_exp = 'reading'
                trial_ctr += 1
                # exclude specific screens of subjects
                if subject_id in exclude_screens and trial_ctr in exclude_screens[subject_id]:
                    select = False
                else:
                    select = True

                # hardcode special case for ET_64, where experiment interrupted and is split in 2 halfs
                # the first part also half includes the interrupted trial, so remove that one
                if filepath.split('/')[2] == 'decoding_v18_deploy' and subject_id == 'ET_64' and trial_ctr == 19:
                    select = False

            elif mp.trial_id.match(line):
                TRIAL_ID = int(line.split()[-1])

            elif mp.trial_index.match(line):
                Trial_Index_ = int(line.split()[-1]) + 1

            elif mp.item_id.match(line):
                item_id = line.split()[-1]

            elif mp.off_reading_text.match(line):
                select = False

            elif mp.on_reading_header.match(line):
                select = True  # won't be written because experiment is 'header' and is not in yaml experiments
                curr_exp = 'header'

            elif mp.off_reading_header.match(line):
                select = False
                curr_exp = 'header'

        # if line is a sample (not a message)
        elif select:
            if curr_exp not in experiments:
                continue
            # m = mp.eye_tracking_sample.match(line)
            if len(line.split('\t')) > 6:
                m = mp.eye_tracking_sample_bino.match(line)
            else:
                m = mp.eye_tracking_sample_mono.match(line)
            if not m:
                continue

            # write recorded samples into dictionary
            for column in columns['sample']:
                value = m.group(column)
                if column == 'time':
                    try:
                        value = int(value)
                    except:
                        logging.error(f'TIMESTAMP COULD NOT BE CASTED AS'
                                      f' INTEGER! Aborting {filename}!')
                        return -1
                else:
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        value = np.nan
                data_out[curr_exp][column].append(value)

            # write experiment specific data into dictionary

            if curr_exp == 'reading':

                # make sure that the idx and ids are correct
                if not subject_id in ['ET_64']:
                    assert TRIAL_ID == idx_to_id_dict[Trial_Index_][0]
                    assert item_id == idx_to_id_dict[Trial_Index_][1]
                    assert trial_ctr == TRIAL_ID

                model = idx_to_id_dict[Trial_Index_][2]
                decoding_strategy = idx_to_id_dict[Trial_Index_][3]

                data_out[curr_exp]['item_id'].append(item_id)
                data_out[curr_exp]['TRIAL_ID'].append(TRIAL_ID)
                data_out[curr_exp]['Trial_Index_'].append(Trial_Index_)
                data_out[curr_exp]['model'].append(model)
                data_out[curr_exp]['decoding_strategy'].append(decoding_strategy)

        else:
            # TODO: Check what types of lines we are discarding here
            pass

    asc_file.close()

    for experiment in experiments:

        logging.info(f'writing {experiment} to {filepath_csv}')

        file_columns = columns[experiment] + columns['sample']
        df = pd.DataFrame(data=data_out[experiment])
        df[file_columns].to_csv(filepath_csv, index=False, sep='\t', na_rep='NaN')

    return 0
