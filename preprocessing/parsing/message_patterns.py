#!/usr/bin/env python3
"""
Message patterns from the eye tracker output. Needed to parse the ascii files and extract relevant information to be
written to the csv files.
"""
import re


# line containing eyetracking sample for both eyes
eye_tracking_sample_bino = re.compile(
    r'(?P<time>\d*)\s+'
    r'(?P<x_left>[-]?\d*[.]\d*)\s+'
    r'(?P<y_left>[-]?\d*[.]\d*)\s+'
    r'(?P<pupil_left>\d*[.]\d*)\s+'
    r'(?P<x_right>[-]?\d*[.]\d*)\s+'
    r'(?P<y_right>[-]?\d*[.]\d*)\s+'
    r'(?P<pupil_right>\d*[.]\d*)\s*'
    r'(?P<extra>0?[.]?0?)\s*'
    r'((?P<dots>[A-Za-z.]{5}))?\s*$',
)

# line containing eyetracking sample --> for only mono eye tracking, which eye does not matter
eye_tracking_sample_mono = re.compile(
    r'(?P<time>\d*)\s+'
    r'(?P<x_right>[-]?\d*[.]\d*)\s+'
    r'(?P<y_right>[-]?\d*[.]\d*)\s+'
    r'(?P<pupil_right>\d*[.]\d*)\s+'
    r'(?P<extra>0?[.]?0?)\s*'
    r'((?P<dots>[A-Za-z.]{3}))?\s*$',
)


on_reading_header = re.compile('.*ONSET_RECORDING_INTRODUCTORY_SENTENCE$')
off_reading_header = re.compile('.*END_RECORDING_INTRODUCTORY_SENTENCE$')

on_reading_text = re.compile(r'.*ONSET_RECORDING_TEXT$')
off_reading_text = re.compile(r'.*END_RECORDING_TEXT$')


trial_id = re.compile(r'.*TRIAL_VAR\s+TRIAL_ID\s+\d+')
trial_index = re.compile(r'.*TRIAL_VAR\s+Trial_Index_\s+\d+')
item_id = re.compile(r'.*TRIAL_VAR\s+item_id\s+item\d+')


# driftcorrect
driftcorrect_any = re.compile(r'MSG\s+(?P<time>\d+)\s+DRIFTCORRECT\s+.*$')

driftcorrect = re.compile(
    r'MSG\s+(?P<time>\d+)\s+DRIFTCORRECT\s+LR\s+'
    r'(?P<eye>LEFT|RIGHT)\s+'
    r'at\s+(?P<x_pos>\d+),(?P<y_pos>\d+)\s+'
    r'OFFSET\s+(?P<offset_deg>[-]?\d+[.]\d+)\s+deg[.]\s+'
    r'(?P<offset_x_pixel>[-]?\d+[.]\d+),'
    r'(?P<offset_y_pixel>[-]?\d+[.]\d+) pix[.].*$',
)

driftcorrect_aborted = re.compile(r'MSG\s+(?P<time>\d+)\s+DRIFTCORRECT\s+LR\s+ABORTED$')

driftcorrect_repeating = re.compile(
    r'MSG\s+(?P<time>\d+)\s+DRIFTCORRECT\s+LR\s+'
    r'REPEATING due to large error\s+'
    r'L=(?P<left_error>[-]?\d+.\d+)\s+'
    r'R=(?P<right_error>[-]?\d+.\d+)\s+'
    r'drift_correction_maxerr=(?P<max_error>[-]?\d+.\d+).*$',
)

# video sync message
videosync = re.compile(
    r'MSG\s+(?P<time_msg>\d+)\s+'
    r'(?P<uuid>[a-f0-9]{32})\s+'
    r'(?P<time_sync>\d+[.]\d+)',
)
