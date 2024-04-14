#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script that reads through the ascii files, extracts the relevant information and writes it into csv files.
"""

import logging
import os

from datetime import datetime
from joblib import Parallel, delayed
from typing import Dict, List, Union
from argparse import ArgumentParser

import numpy as np

from preprocessing.utils.loading import load_config
from preprocessing.parsing.parse_asc import parse_asc_file

import glob


logging.basicConfig(format='%(levelname)s::%(message)s',
                    level=logging.INFO)



def process_asc_to_csv(path_to_data: str,
                       experiments: List[str],
                       columns: Union[str, Dict[str, str]],
                       exclude_subjects: List[str],
                       exclude_screens: Dict[str, List[int]],
                       n_jobs: int = 1,
                       check_file_exists: bool = True):

    subj_dirs = glob.glob(os.path.join(path_to_data, '*'))

    # exclude entire subjects
    asc_files_paths = [
        os.path.join(subj_dir, f'{subj_dir.split("/")[-1]}.asc')
        for subj_dir in subj_dirs
        if f'{subj_dir.split("/")[-1]}' not in exclude_subjects
    ]

    # filenames = [filename for filename in os.listdir(path_asc_files)
    #              if os.path.splitext(filename)[1] == '.asc']
    # filenames.sort()
    logging.info(f'Input files ({len(asc_files_paths)}): {asc_files_paths}')

    # parse files in parallel
    done = Parallel(n_jobs=n_jobs)(
        delayed(parse_asc_file)(filepath=asc_file_path,
                                experiments=experiments,
                                columns=columns,
                                exclude_screens=exclude_screens,
                                check_file_exists=check_file_exists)
        for asc_file_path in asc_files_paths)

    # parse files sequentially
    # for asc_file_path in asc_files_paths:
    #     done = parse_asc_file(
    #         filepath=asc_file_path,
    #         experiments=experiments,
    #         columns=columns,
    #         exclude_screens=exclude_screens,
    #         check_file_exists=check_file_exists,
    #     )

    return 0


def main():
    start_time = datetime.now()

    config = load_config()

    path_to_data = 'data/'
    experiments = config['asc2csv']['experiments']
    columns = config['asc2csv']['columns']
    n_jobs = config['asc2csv']['n_jobs']
    check_file_exists = config['asc2csv']['check_file_exists']
    exclude_subjects = config['exclude']['subjects']
    exclude_screens = config['exclude']['screens']

    process_asc_to_csv(
        path_to_data=path_to_data,
        experiments=experiments,
        columns=columns,
        exclude_subjects=exclude_subjects,
        exclude_screens=exclude_screens,
        n_jobs=n_jobs,
        check_file_exists=check_file_exists,
    )

    logging.info(f'Took {datetime.now() - start_time}')



if __name__ == "__main__":
    raise SystemExit(main())
