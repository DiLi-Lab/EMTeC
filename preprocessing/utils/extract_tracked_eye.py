#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List, Dict
import codecs
import os
import pickle

"""
Script to extract which eye was being tracked from the ASCII files.
"""


def extract_tracked_eye(
        subject_ids: List[str],
        path_to_subjects: str,
) -> Dict[str, str]:
    
    eyes_dict = dict()
    for subject in subject_ids:

        print(f'Extracting eye tracked for {subject}')

        # open the ASCII file
        subj_dir = os.path.join(path_to_subjects, subject)
        asc_filename = os.path.join(subj_dir, subject + '.asc')
        asc_file = codecs.open(asc_filename, 'r', encoding='ascii', errors='ignore')

        # read through the file line by line until first line with info on eye tracked
        line = True
        while line:
            line = asc_file.readline()
            if line.startswith('EVENTS'):
                split_line = line.split('\t')  
                if split_line[0] == 'EVENTS' and split_line[1] == 'GAZE':
                    eyes_dict[subject] = split_line[2]
                    break
        
    return eyes_dict
