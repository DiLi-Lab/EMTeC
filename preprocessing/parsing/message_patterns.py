#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Message patterns from the eye tracker output. Needed to parse the ascii files and extract relevant information to be
written to the csv files.
"""

import re


# line containing eyetracking sample for both eyes
eye_tracking_sample_bino = re.compile(r'(?P<time>\d*)\s+'
                                 r'(?P<x_left>[-]?\d*[.]\d*)\s+'
                                 r'(?P<y_left>[-]?\d*[.]\d*)\s+'
                                 r'(?P<pupil_left>\d*[.]\d*)\s+'
                                 r'(?P<x_right>[-]?\d*[.]\d*)\s+'
                                 r'(?P<y_right>[-]?\d*[.]\d*)\s+'
                                 r'(?P<pupil_right>\d*[.]\d*)\s*'
                                 r'(?P<extra>0?[.]?0?)\s*'
                                 r'((?P<dots>[A-Za-z.]{5}))?\s*$')

# line containing eyetracking sample --> for only mono eye tracking, which eye does not matter
eye_tracking_sample_mono = re.compile(r'(?P<time>\d*)\s+'
                                 r'(?P<x_right>[-]?\d*[.]\d*)\s+'
                                 r'(?P<y_right>[-]?\d*[.]\d*)\s+'
                                 r'(?P<pupil_right>\d*[.]\d*)\s+'
                                 r'(?P<extra>0?[.]?0?)\s*'
                                 r'((?P<dots>[A-Za-z.]{3}))?\s*$')



# line containing eyetracking sample --> for only left eye tracking
# eye_tracking_sample = re.compile(r'(?P<time>\d*)\s+'
#                                  r'(?P<x_left>[-]?\d*[.]\d*)\s+'
#                                  r'(?P<y_left>[-]?\d*[.]\d*)\s+'
#                                  r'(?P<pupil_left>\d*[.]\d*)\s+'
#                                  r'(?P<extra>0?[.]?0?)\s*'
#                                  r'((?P<dots>[A-Za-z.]{3}))?\s*$')

# line containing eyetracking sample --> for only right eye tracking
# eye_tracking_sample = re.compile(r'(?P<time>\d*)\s+'
#                                  r'(?P<x_right>[-]?\d*[.]\d*)\s+'
#                                  r'(?P<y_right>[-]?\d*[.]\d*)\s+'
#                                  r'(?P<pupil_right>\d*[.]\d*)\s+'
#                                  r'(?P<extra>0?[.]?0?)\s*'
#                                  r'((?P<dots>[A-Za-z.]{3}))?\s*$')


## READING regex patterns
# on_reading = re.compile('.*SYNCTIME_READING$')
# off_reading = re.compile('.* READING[.]STOP$')
#
# # on_reading_trial = re.compile('.*READING_TRIAL_ID [1-4]$') # doesn't exist (originally for sleepalc)
# off_reading_trial = re.compile('.*TRIAL_READING[.]STOP$') # 1 per text
#
# # on_reading_header = re.compile('.*SYNCTIME_HEADER$')
# # off_reading_header = re.compile('.*HEADER[.]STOP$')
on_reading_header = re.compile('.*ONSET_RECORDING_INTRODUCTORY_SENTENCE$')
off_reading_header = re.compile('.*END_RECORDING_INTRODUCTORY_SENTENCE$')

# on_screen = re.compile(r'.*SYNCTIME_READING_SCREEN_(?P<screen_id>[1-9]\d*)$')
# off_screen = re.compile(r'.*READING_SCREEN_(?P<screen_id>[1-9]\d*)[.]STOP$')
on_reading_text = re.compile(r'.*ONSET_RECORDING_TEXT$')
off_reading_text = re.compile(r'.*END_RECORDING_TEXT$')

# on_question = re.compile(r'.*SYNCTIME_[Q]?(?P<question_id>[1-9]\d*)$')
# off_question = re.compile(r'.*Q(?P<question_id>[1-9]\d*)[.]STOP$')

#text_id = re.compile(r'MSG\s+(?P<timestamp>\d+) !V TRIAL_VAR textid (?P<text_id>\d+)')
# trial_id = re.compile(r'MSG\s+(?P<timestamp>\d+) !V TRIAL_VAR READING_TRIAL_ID (?P<trial_id>\d+)')

trial_id = re.compile(r'.*TRIAL_VAR\s+TRIAL_ID\s+\d+')
trial_index = re.compile(r'.*TRIAL_VAR\s+Trial_Index_\s+\d+')
item_id = re.compile(r'.*TRIAL_VAR\s+item_id\s+item\d+')

# def on_question_id(question_id: int) -> re.Pattern:
#     if question_id == 10:
#         return re.compile('.*SYNCTIME_[Q]?10$')
#     else:
#         return re.compile(f'.*SYNCTIME_[Q]?{question_id}$')
#
# def off_question_id(question_id: int) -> re.Pattern:
#     return re.compile(f'.*Q{question_id}[.]STOP$')
#
# def on_screen_id(screen_id: int) -> re.Pattern:
#     return re.compile(f'.*SYNCTIME_READING_SCREEN_{screen_id}$')
#
# def off_screen_id(screen_id: int) -> re.Pattern:
#     return re.compile(f'.*READING_SCREEN_{screen_id}[.]STOP$')
#
#
# # KAROLINSKA regex patterns
# on_karolinska = re.compile(r'MSG\s+(?P<time>\d+)\s+[-]?\d+'
#                            r'\s+SYNCTIME_KAROLINSKA$')
# off_karolinska = re.compile(r'MSG\s+(?P<time>\d+)'
#                             r'\s+\d\s+KAROLINSKA[.]STOP$')


# driftcorrect
driftcorrect_any = re.compile(r'MSG\s+(?P<time>\d+)\s+DRIFTCORRECT\s+.*$')

driftcorrect = re.compile(r'MSG\s+(?P<time>\d+)\s+DRIFTCORRECT\s+LR\s+'
                          r'(?P<eye>LEFT|RIGHT)\s+'
                          r'at\s+(?P<x_pos>\d+),(?P<y_pos>\d+)\s+'
                          r'OFFSET\s+(?P<offset_deg>[-]?\d+[.]\d+)\s+deg[.]\s+'
                          r'(?P<offset_x_pixel>[-]?\d+[.]\d+),'
                          r'(?P<offset_y_pixel>[-]?\d+[.]\d+) pix[.].*$')

driftcorrect_aborted = re.compile(r'MSG\s+(?P<time>\d+)\s+DRIFTCORRECT\s+LR\s+ABORTED$')

driftcorrect_repeating = re.compile(r'MSG\s+(?P<time>\d+)\s+DRIFTCORRECT\s+LR\s+'
                                    r'REPEATING due to large error\s+'
                                    r'L=(?P<left_error>[-]?\d+.\d+)\s+'
                                    r'R=(?P<right_error>[-]?\d+.\d+)\s+'
                                    r'drift_correction_maxerr=(?P<max_error>[-]?\d+.\d+).*$')

# video sync message
videosync = re.compile(r'MSG\s+(?P<time_msg>\d+)\s+'
                       r'(?P<uuid>[a-f0-9]{32})\s+'
                       r'(?P<time_sync>\d+[.]\d+)')
