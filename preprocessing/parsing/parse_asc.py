#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Eyetracking-Data extraction
Experiments: Reading
This script extracts the lines with samples (timestamp, x-screen coordinate, y-screen coordinate, pupil diameter)
from Eyelink 100 raw data files (previously converted from edf to ascii).

Extract recording samples on experimental trials;
remove samples on practice trials and on game trials;
remove samples between trials (calibration)

Author: Lena Jäger
Edited by: Patrick Haller (October 2020),
           Daniel Krakowczyk (November 2020)
'''
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


# from utils.naming import create_filepath_for_csv_file
# from utils.naming import get_subject_id, get_session_id


# def check_file(filename):
#     # files to exclude from parsing (e.g., aborted sessions):
#     black_list = [#'ET_003_1.asc', 'ET_010_4.asc', 'ET_022_2.asc', 'ET_023_3.asc', 'ET_023_4.asc', 'ET_029_3.asc',
#                   #'ET_036_2.asc', 'ET_056_2.asc', 'ET_057_1.asc', 'ET_065_4.asc',
#                   # 'ET_020_4.asc', 'ET_021_4.asc', 'ET_025_1.asc', 'ET_025_3.asc', 'ET_027_1.asc', 'ET_027_4.asc', 'ET_028_2.asc'
#                 ] # add all files that were aborted but repeated under a different file name
#
#     if filename in black_list:
#         logging.info(f'Parsing of file {filename} was not completed because file is on black list (e.g., aborted session).')
#         return 0
#     # handling mistakes in file names
#     # elif re.compile('0057-.*.asc').match(file):  # recode subject_id 57 followed by hyphen as subject_id 0 (subject_id 57_ already exists)
#     #    subject_id = 0
#     #    session_id = int(file.split('-')[1].split('.')[0])  # session was coded correctly
#
#     # treat cases [0-9]^4-[0-9].asc and [0-9]^4_[0-9]_[0-9].asc
#     elif len(filename.split('_')) == 2:
#         subject_id = int(filename.split('_')[0]) # reader id
#         session_id = (filename.split('_')[1].split('.')[0])# session
#     elif len(filename.split('_')) == 1 and len(filename.split('-')) == 2:
#         subject_id = int(filename.split('-')[0])
#         session_id = (filename.split('-')[1].split('.')[0])
#     elif len(filename.split('_')) == 3:
#         subject_id = int(get_subject_id(filename))
#         session_id = int(get_session_id(filename))
#     else:
#         logging.error(f'unknown file name format: {filename}. could not extract subject and session id.')
#         return 0
#
#     return (subject_id, session_id)


def parse_asc_file(
        filepath: str,  # path to the asc file, e.g., './data/decoding_v17_deploy/results/ET_46/ET_46.asc'
        experiments: List[str],
        columns: Union[str, Dict[str, str]],
        exclude_screens: Dict[str, List[int]],
        check_file_exists: bool = True):
    filename = filepath.split('/')[-1]
    subject_id = filename[:-4]
    filepath_csv = filepath[:-4] + '.csv'
    logging.basicConfig(format='%(levelname)s::%(message)s',
                        level=logging.INFO)
    logging.info(f'parsing file {filename}')

    # if check_file(filename) == 0:
    #     return 0
    # else:
    #     (subject_id, session_id) = check_file(filename)

    # Skip if all csv files exist
    csv_file_exists = []

    if check_file_exists:
        # for experiment in experiments:
        #     filepath = create_filepath_for_csv_file(
        #         basepath=path_csv_files,
        #         subject_id=subject_id,
        #         session_id=session_id,
        #         experiment_type=experiment)

        csv_file_exists.append(os.path.isfile(filepath_csv))

    if all(csv_file_exists):
        logging.info(f'all csv files for {filename} exist. skipping.')
        return 0

    # reading
    # reading_screen = -1
    # # reading_question = -1
    # reading_trial_id = 0
    # reading_text_id = -1
    print(filepath)

    item_id = -1
    TRIAL_ID = -1
    Trial_Index_ = -1
    trial_ctr = 0

    # # karolinska
    # karo_block_id = 0

    # STORING
    data_out = defaultdict(lambda: defaultdict(list))
    select = False  # true if next eyetracking sample should be written out
    curr_exp = ''  # current experiment to write sample for
    drift_corrected = False

    # asc_file = codecs.open(os.path.join(path_asc_files, filename),
    #                        'r', encoding='ascii', errors='ignore')
    asc_file = codecs.open(filepath, 'r', encoding='ascii', errors='ignore')

    # load dictionary that maps Trial_Index_ to TRIAL_ID and item_id as sanity check
    path_to_mapping_dict = filepath[:-4] + '_idx_to_id.pickle'
    with open(path_to_mapping_dict, 'rb') as pickle_file:
        idx_to_id_dict = pickle.load(pickle_file)

    line = True
    #  questions = False  # TODO remove hard-coding
    while line:
        try:
            line = asc_file.readline()
        except UnicodeDecodeError:
            logging.error(f'DECODING ERROR, aborting file {filename}')
            return -1

        if line.startswith('MSG'):
            ## READING

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
                # if subject_id in exclude_screens and TRIAL_ID in exclude_screens[subject_id]:
                #     breakpoint()
                #     select = False

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

            # if mp.on_reading.match(line):
            #     curr_exp = 'reading'
            #     select = False
            #     reading_screen = 0
            #     reading_question = -1
            #
            # elif mp.on_reading_header.match(line):
            #     curr_exp = 'reading'
            #     # set to next trial index
            #     reading_trial_id += 1
            #     reading_question = -1
            #
            # elif mp.on_screen.match(line):
            #     select = True
            #     m = mp.on_screen.match(line)
            #     reading_screen = int(m.group('screen_id'))
            #
            # elif mp.off_reading_trial.match(line):
            #     # screen_id = 0 represents period between two reading trials
            #     reading_screen = 0
            #     select = False
            #
            # elif mp.off_screen.match(line):
            #     # curr_exp = 'NA'
            #     select = False
            #     reading_screen = 0
            #
            # elif mp.on_question.match(line) and questions:
            #     curr_exp = 'questions'
            #     select = True
            #     m = mp.on_question.match(line)
            #     reading_question = int(m.group('question_id'))
            #
            # elif mp.on_question_id(10).match(line) and questions:
            #     curr_exp = 'questions'
            #     select = True
            #     reading_question = 10
            #
            # elif mp.off_question.match(line) and questions:
            #     curr_exp = 'NA'
            #     reading_question = -1
            #     select = False
            #
            # elif mp.off_reading.match(line):
            #     curr_exp = 'NA'
            #     select = False
            #
            # elif mp.text_id.match(line):
            #     m = mp.text_id.match(line)
            #     reading_text_id = int(m.group('text_id'))

            # # TODO CHECK Karolinska Block inbetween PVT and Reading
            # elif mp.on_karolinska.match(line):
            #         select = False
            #         curr_exp = 'karolinska'
            #         karo_block_id += 1
            #         # some block ids are wrongly skipped in actual experiment
            #         if karo_block_id in [12, 14, 16]:
            #             karo_block_id += 1
            #
            # elif mp.off_karolinska.match(line):
            #     select = False
            #     curr_exp = 'NA'

            # TODO check what this does -- do I need it for decoding? what is an unexpected drifftcorrect?
            # mandatory driftcorrects
            # elif mp.driftcorrect.match(line):
            #     drift_corrected = True
            #     check_driftcorrect = (curr_exp == 'reading' and reading_screen == 0)
            #
            #     if check_driftcorrect:
            #         time = mp.driftcorrect.match(line).group('time')
            #         if select:
            #             logging.error(f'UNEXPECTED DRIFTCORRECT OCCURED'
            #                           f' DURING TRIAL AT TIMESTEP {time}!'
            #                           f' Aborting {filename}!')
            #             return -1

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
            #   data_out[curr_exp]['trial_ctr'].append(trial_ctr)

            # data_out[curr_exp]['trial_id'].append(reading_trial_id)
            # data_out[curr_exp]['screen_id'].append(reading_screen)
            #
            # # data_out[curr_exp]['question_id'].append(reading_question)
            # data_out[curr_exp]['text_id'].append(reading_text_id)

            # elif curr_exp == 'karolinska':
            #     data_out[curr_exp]['block_id'].append(karo_block_id)

        else:
            # TODO: Check what types of lines we are discarding here
            pass

    asc_file.close()

    for experiment in experiments:
        # filepath = create_filepath_for_csv_file(
        #     basepath=path_csv_files,
        #     subject_id=subject_id,
        #     session_id=session_id,
        #     experiment_type=experiment)

        logging.info(f'writing {experiment} to {filepath_csv}')

        file_columns = columns[experiment] + columns['sample']
        df = pd.DataFrame(data=data_out[experiment])
        df[file_columns].to_csv(filepath_csv, index=False, sep='\t', na_rep='NaN')

    return 0
