#!/usr/bin/env python3
"""
Code for preprocessing raw samples into fixation and saccade events
Coding: fixation=1, saccade=2, corrupt=3
"""
from __future__ import annotations

import glob
import logging
import os
import pickle
import sys
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel

from preprocessing.parsing.data_analysis import extract_fixation_means
from preprocessing.parsing.ed_helpers import compute_peak_velocity_and_amplitude
from preprocessing.parsing.ed_helpers import corruptSamplesIdx
from preprocessing.parsing.ed_helpers import estimate_threshold
from preprocessing.parsing.ed_helpers import microsacc
from preprocessing.parsing.ed_helpers import pix2deg
from preprocessing.parsing.ed_helpers import vecvel
from preprocessing.parsing.map_fix_to_roi import get_aois_from_event_data
from preprocessing.utils.loading import load_config
from preprocessing.utils.naming import create_filepath_for_event_file
from preprocessing.utils.plotting import plot_ampl_over_vel
from preprocessing.utils.plotting import plot_px_over_time


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--disable-parallel',
        action='store_true',
        help='if provided, fixation extraction is not executed in parallel'
    )
    parser.add_argument(
        '--plot-px-time',
        action='store_true',
        help='if provided, the coordinates over time and the predicted fixations are plotted for every text.',
    )
    parser.add_argument(
        '--plot-ampl-vel',
        action='store_true',
        help='if provided, the saccade amplitude is plotted against peak velocity for every text.',
    )
    parser.add_argument(
        '--threshold',  # Tuple[int, int], subject_based, trial_based
        help='the threshold to use or how to estimate it.',
        default='trial_based',
    )
    parser.add_argument(
        '--threshold-factor',
        type=float,
        default=3,
        help='the factor with which the threshold (standard deviation of the velocity distribution) is multiplied to obtain the radius.'
    )
    parser.add_argument(
        '--threshold-method',
        type=str,
        default='engbert2015',
        choices=['std', 'mad', 'engbert2003', 'engbert2015']
    )
    parser.add_argument(
        '--min-fixation-duration-ms',
        type=int,
        default=20,
        help='the minimum fixation threshold in ms',
    )
    parser.add_argument(
        '--min-saccade-duration-ms',
        type=int,
        default=6,
        help='the minimum saccade duration in ms',
    )
    parser.add_argument(
        '--max-saccade-velocity',
        type=int,
        default=500,
        help='the maximum saccade velocity in degrees/s.',
    )
    parser.add_argument(
        '--theta',
        type=float,
        default=0.6,
        help='velocity threshold in degrees/s',
    )
    return parser


def group_df(df: pd.DataFrame):
    grouped = df.groupby('item_id')
    sub_dfs = list()
    for item_id, group in grouped:
        sub_dfs.append(group)
    return sub_dfs


def process_csv_to_events(
        path_to_subjects: str,
        columns: str | dict[str, str],
        exclude_subjects: list[str],
        n_jobs: int = 1,
        threshold: str = 'trial_based',
        threshold_factor: int = 3,
        threshold_method: str = 'engbert2015',
        min_fixation_duration_ms: int = 20,
        min_saccade_duration_ms: int = 6,
        max_saccade_velocity: int = 600,
        theta: float = 0.6,
        eye: str = 'right',
        unit: str = 'px',
        disable_parallel: bool = True,
        plot_px_time: bool = False,
        plot_ampl_velocity: bool = False,
        check_file_exists: bool = True,
) -> int:

    subj_dirs = glob.glob(os.path.join(path_to_subjects, '*'))
    logging.info(f'Input files ({len(subj_dirs)}): {subj_dirs}')

    # experiment setup
    expt = Experiment(screenPX_x=1280, screenPX_y=1024, screenCM_x=38.2, screenCM_y=30.2, dist=60, sampling=2000)

    if disable_parallel:
        for subj_dir in subj_dirs:
            event_data = readfile_event(
                Experiment_Class=expt,
                subj_dir=subj_dir,
                columns=columns,
                exclude_subjects=exclude_subjects,
                threshold=threshold,
                threshold_factor=threshold_factor,
                threshold_method=threshold_method,
                min_fixation_duration_ms=min_fixation_duration_ms,
                min_saccade_duration_ms=min_saccade_duration_ms,
                max_saccade_velocity=max_saccade_velocity,
                theta=theta,
                eye=eye,
                unit=unit,
                start_time=False,
                down_sampling_factor=1,
                plot_px_time=plot_px_time,
                plot_ampl_velocity=plot_ampl_velocity,
            )
    else:
        event_data = Parallel(n_jobs=n_jobs)(
            delayed(readfile_event)(
                Experiment_Class=expt,
                subj_dir=subj_dir,
                columns=columns,
                exclude_subjects=exclude_subjects,
                threshold=threshold,
                threshold_factor=threshold_factor,
                threshold_method=threshold_method,
                min_fixation_duration_ms=min_fixation_duration_ms,
                min_saccade_duration_ms=min_saccade_duration_ms,
                max_saccade_velocity=max_saccade_velocity,
                theta=theta,
                eye=eye,
                unit=unit,
                start_time=False,
                down_sampling_factor=1,
                plot_px_time=plot_px_time,
                plot_ampl_velocity=plot_ampl_velocity,
            )
            for subj_dir in subj_dirs
        )

    return 0


# Experiment configuration:
class Screen:  # properties of the screen
    def __init__(
        self,
        screenPX_x: int | float,
        screenPX_y: int | float,
        screenCM_x: int | float,
        screenCM_y: int | float,
        dist: int | float,
    ):
        self.px_x = screenPX_x  # width of the screen in pixels (resolution)
        self.px_y = screenPX_y  # height of the screen in pixels (resolution)
        self.cm_x = screenCM_x  # width of the screen in cm
        self.cm_y = screenCM_y  # height of the screen in cm
        # maximal/minimal screen coordinates in degrees of visual angle:
        self.x_max = pix2deg(screenPX_x - 1, screenPX_x, screenCM_x, dist)
        self.y_max = pix2deg(screenPX_y - 1, screenPX_y, screenCM_y, dist)
        self.x_min = pix2deg(0, screenPX_x, screenCM_x, dist)
        self.y_min = pix2deg(0, screenPX_y, screenCM_y, dist)


class Experiment:
    def __init__(
            self,
            screenPX_x: int | float,  # width of the screen in pixels (resolution)
            screenPX_y: int | float,  # height of the screen in pixels (resolution)
            screenCM_x: int | float,  # width of the screen in cm
            screenCM_y: int | float,  # height of the screen in cm
            dist: int | float,        # distance between eyes and the screen
            sampling: int | float,    # the sampling rate
    ):
        self.sampling = sampling  # sampling rate in Hz
        self.dist = dist  # eye-to-screen distance in cm
        self.screen = Screen(screenPX_x, screenPX_y, screenCM_x, screenCM_y, dist)


# = process_asc_to_csv
def readfile_event(
        Experiment_Class: Experiment,
        subj_dir: str,
        columns: str | dict[str, str],
        exclude_subjects: list[str],
        threshold: str = 'trial_based',
        threshold_factor: int = 3,
        threshold_method: str = 'engbert2015',
        min_fixation_duration_ms: int = 20,
        min_saccade_duration_ms: int = 6,
        max_saccade_velocity: int = 600,
        theta: float = 0.6,
        eye: str = 'right',
        unit: str = 'px',
        start_time: bool = False,
        down_sampling_factor: int = 1,
        trial_based_time: bool = False,
        check_file_exists: bool = True,
        plot_px_time: bool = False,
        plot_ampl_velocity: bool = False,
):
    """
    read file to get one event per line
    """
    subj_id = subj_dir.split('/')[-1]

    if subj_id in exclude_subjects:
        print(f'---excluding subject {subj_id}')
        return


    csv_filepath = os.path.join(subj_dir, f'{subj_id}.csv')
    event_dir = os.path.join(subj_dir, 'fixations')

    if not os.path.exists(event_dir):
        os.makedirs(event_dir)

    # save the program call commands
    command_args = ' '.join(sys.argv)
    with open(os.path.join(event_dir, 'command_log.txt'), 'a') as f:
        f.write(command_args + '\n')

    # load dictionary that maps from the Trial_Index_ to the item_id, TRIAL_ID, model, and decoding strategy
    with open(os.path.join(subj_dir, f'{subj_id}_idx_to_id.pickle'), 'rb') as pickle_file:
        mapping_dict = pickle.load(pickle_file)

    conversion_factor = Experiment_Class.sampling / 1000
    min_fixation_duration_samples = min_fixation_duration_ms * conversion_factor
    min_saccade_duration_samples = min_saccade_duration_ms * conversion_factor

    print(f'preprocessing of file {csv_filepath}')

    logging.basicConfig(
        format='%(levelname)s::%(message)s',
        level=logging.INFO,
    )

    logging.info(f'parsing file {csv_filepath}')

    # select columns to be included:
    coords = []  # columns containing x/y coordinates for the selected eye(s)
    if eye in ['right', 'both']:
        #coords = coords + ['x_right', 'y_right', 'time']
        coords = coords + ['x', 'y', 'time']
    if eye in ['left', 'both']:
        #coords = coords + ['x_left', 'y_left', 'time']
        coords = coords + ['x', 'y', 'time']

    csv_columns = columns['reading'] + coords

    d = pd.read_csv(csv_filepath, usecols=csv_columns, delimiter='\t', na_values=[''])

    # get all the filenames; first get all trial ids etc
    unique_item_ids = d['item_id'].unique()
    unique_Trial_Index_ = d['Trial_Index_'].unique()
    unique_TRIAL_ID = d['TRIAL_ID'].unique()

    zipped_info = list()
    for u_item_id, u_trial_index, u_trial_id in zip(unique_item_ids, unique_Trial_Index_, unique_TRIAL_ID):
        model = mapping_dict[u_trial_index][2]
        decoding_strategy = mapping_dict[u_trial_index][3]
        zipped_info.append((u_item_id, u_trial_index, u_trial_id, model, decoding_strategy))

    event_file_exists = []

    if check_file_exists:
        for item_id, Trial_Index_, TRIAL_ID, model, decoding_strategy in zipped_info:
            filepath = create_filepath_for_event_file(
                subj_id=subj_id,
                event_dir=event_dir,
                item_id=item_id,
            )
            event_file_exists.append(os.path.isfile(filepath))

    if all(event_file_exists):
        logging.info(f'all fixation csv files for {csv_filepath} exist. skipping.')
        return 0

    assert unit in ['velocity', 'px', 'deg']

    # add subject id to file
    d['subject_id'] = subj_id

    sampling = Experiment_Class.sampling / down_sampling_factor

    # add columns for coordinates in degrees of visual angle
    deg_cols = []
    for coord in coords:
        d[coord + '_deg'] = pix2deg(d[coord], Experiment_Class.screen.px_x, Experiment_Class.screen.cm_x,
                                    Experiment_Class.dist)
        deg_cols.append(coord + '_deg')

    # add columns for gaze velocities
    vel_cols = []  # columns with velocity dx_deg, dy_deg
    for deg_col in deg_cols:
        d['d' + deg_col] = d[deg_col] - d[deg_col].shift(1)
        vel_cols.append('d' + deg_col)


    ## columns are already named in a non-specific way (x and y instead of x_right, y_right and x_left, y_left)
    # # Start preprocessing, choose selected eye
    # if eye == 'both':  # to be done: implement binocular preprocessing: see Engbert's microsaccade detection
    #     print('Binocular preprocessing is not defined yet')
    #     return -1
    # if eye == 'left':
    #     choose_data_from_left(d)
    # else:
    #     choose_data_from_right(d)
    # rename time column

    d['t'] = d['time']

    # split the dataframe into the individual texts
    sub_dfs = group_df(d)

    for sub_df in sub_dfs:

        assert len(sub_df['item_id'].unique()) == 1, "The column item_id doesn't have only one unique value."
        assert len(sub_df['TRIAL_ID'].unique()) == 1, "The column TRIAL_ID doesn't have only one unique value."
        assert len(sub_df['Trial_Index_'].unique()) == 1, "The column Trial_Index_ doesn't have only one unique value."
        assert len(sub_df['model'].unique()) == 1, "The column model doesn't have only one unique value."
        assert len(sub_df['decoding_strategy'].unique()) == 1, "The column decoding_strategy doesn't have only one unique value."

        print(f'--- processing trial id {TRIAL_ID}')

        item_id = sub_df['item_id'].unique().item()
        TRIAL_ID = sub_df['TRIAL_ID'].unique().item()
        Trial_Index_ = sub_df['Trial_Index_'].unique().item()
        model = sub_df['model'].unique().item()
        decoding_strategy = sub_df['decoding_strategy'].unique().item()

        # estimate the velocity threshold
        # if the given threshold is a tuple, use this as the threshold
        if isinstance(threshold, tuple):
            estimated_threshold = threshold
        # else if it is a string, estimate the threshold from the data according to a specified method
        elif isinstance(threshold, str):

            if threshold == 'trial_based':
                # compute velocities based on current trial
                v_est = vecvel(
                    x=np.array(sub_df[['x_deg', 'y_deg']]),
                    sampling_rate=sampling,
                )
            elif threshold == 'subject_based':
                # compute velocities based on all trials of current subject
                v_est = vecvel(
                    x=np.array(d[['x_deg', 'y_deg']]),
                    sampling_rate=sampling,
                )
            else:
                raise ValueError(f'Threshold {threshold} not implemented.')

            # compute the treshold from the given velocities according to a certain method
            estimated_threshold = estimate_threshold(
                velocities=v_est,
                threshold_method=threshold_method,
            )

        # saccade detection for selected eye, add bool column sac, indicating a saccade
        saccade_info, saccs, _ = microsacc(
            velocities=np.array(sub_df[['x_deg', 'y_deg']]),
            threshold=estimated_threshold,
            threshold_factor=threshold_factor,
            min_duration=min_saccade_duration_samples,
            sampling_rate=sampling,
        )
        sub_df['sac'] = saccs

        # identify corrupt samples
        corruptIdx = corruptSamplesIdx(
            sub_df.x_deg, sub_df.y_deg,
            x_max=Experiment_Class.screen.x_max,
            y_max=Experiment_Class.screen.y_max,
            x_min=Experiment_Class.screen.x_min,
            y_min=Experiment_Class.screen.y_min,
            theta=theta,
            samplingRate=sampling,
        )

        # add bool column to indicate corrupt samples
        sub_df['corrupt'] = sub_df.index.isin(corruptIdx)
        # code NaNs as corrupt events
        sub_df['corrupt'] = np.where(np.isnan(sub_df.dx_deg), True, sub_df.corrupt)

        sub_df['event'] = np.where(sub_df.corrupt, 3, np.where(sub_df.sac, 2, 1))  # fix=1, saccade=2, corrupt=3

        event = None  # flag for current event: fix=1; sac=2, corrupt=3
        first = True  # first line of new file
        # dataframe where one row contains one event (sequence of samples)
        if unit == 'velocity':
            # provide cols in alphabetical order
            event_dat = pd.DataFrame(columns=sorted(set(columns['sample_velocity'] + columns['reading'])))
        elif unit == 'deg':
            event_dat = pd.DataFrame(columns=sorted(set(columns['sample_deg'] + columns['reading'])))
        else:
            event_dat = pd.DataFrame(columns=sorted(set(columns['sample_px'] + columns['reading'])))

        # iterate over all samples starting from line 2 (x_deg is nan in first row)
        for index, row in sub_df.iloc[1:].iterrows():
            if event != row['event']:  # onset of new event and event is not first event
                if first:
                    first = False  # if beginning of file, no previous event to write
                else:  # write previous event to dataframe
                    # minimal fixation/saccade duration
                    # after having removed the corrupt samples, event length can be very short;
                    # ---> re-label too short events as corrupt
                    if (event == 1 and len(x_px) < min_fixation_duration_samples) or (
                            event == 2 and len(x_px) < min_saccade_duration_samples
                    ):
                        event = 3
                    # write event to file (also corrupt event)
                    #event_dat = write_event_to_file(event, event_dat, row, unit, x, y, t)
                    event_dat = write_event_to_file_all(event, event_dat, row, x_ddeg, y_ddeg, x_deg, y_deg, x_px, y_px, t)

                # create new event
                event = row['event']  # update event
                # x, y, t = update_xy_event(row, unit)
                x_ddeg, y_ddeg, x_deg, y_deg, x_px, y_px, t = update_xy_event_all(row)
            else:  # event continues
                # continue_xy_event(row, unit, x, y, t)
                continue_xy_event_all(row, x_ddeg, y_ddeg, x_deg, y_deg, x_px, y_px, t)

        # end of file: write the last event
        # re-label too short events as corrupt
        if (event == 1 and len(x_px) < min_fixation_duration_samples) or (
            event == 2 and len(x_px) < min_saccade_duration_samples
        ):
            event = 3
        event_dat = write_event_to_file_all(event, event_dat, row, x_ddeg, y_ddeg, x_deg, y_deg, x_px, y_px, t)

        # compute the peak velocities and amplitudes of the saccades
        # the fixations and corrupt events are coded as -1
        # we need this info to then label too fast saccades as corrupt
        peak_velocities, amplitudes = compute_peak_velocity_and_amplitude(
            event_dat=event_dat,
            sampling=sampling,
        )
        event_dat['peak_velocity'] = peak_velocities
        event_dat['amplitude'] = amplitudes

        for index, row in event_dat.iterrows():
            if row['event'] == 2:
                # if a saccade has a greater peak velocity than 650 degrees/ms: mark it as corrupt
                if row['peak_velocity'] > max_saccade_velocity:
                    event_dat.at[index, 'event'] = 3
        saccade_df = event_dat[event_dat['event'] == 2]

        # plot the processed saccades
        if plot_ampl_velocity:
            plot_ampl_over_vel(
                event_dat=saccade_df,
                sampling=sampling,
                subj_id=subj_id,
                event_dir=event_dir,
                TRIAL_ID=TRIAL_ID,
                item_id=item_id,
                Trial_Index_=Trial_Index_,
                model=model,
                decoding_strategy=decoding_strategy,
            )

        if unit == 'velocity':
            event_dat = event_dat.drop(columns=['seq_x_deg', 'seq_y_deg', 'seq_x', 'seq_y'])
            event_dat['event_len'] = event_dat.seq_dx_deg.str.len()  # length of each event (number of samples)
            event_dat['t_start'] = event_dat.seq_t.apply(lambda x: x[0] if x else None)
            event_dat['t_end'] = event_dat.seq_t.apply(lambda x: x[-1] if x else None)
        elif unit == 'deg':
            event_dat = event_dat.drop(columns=['seq_dx_deg', 'seq_dy_deg', 'seq_x', 'seq_y'])
            event_dat['event_len'] = event_dat.seq_x_deg.str.len()  # length of each event (number of samples)
            event_dat['t_start'] = event_dat.seq_t.apply(lambda x: x[0] if x else None)
            event_dat['t_end'] = event_dat.seq_t.apply(lambda x: x[-1] if x else None)
        else:  # looking at px
            event_dat = event_dat.drop(columns=['seq_dx_deg', 'seq_dy_deg', 'seq_x_deg', 'seq_y_deg'])
            event_dat['event_len'] = event_dat.seq_x.str.len()  # length of each event (number of samples)
            event_dat['t_start'] = event_dat.seq_t.apply(lambda x: x[0] if x else None)
            event_dat['t_end'] = event_dat.seq_t.apply(lambda x: x[-1] if x else None)

        # Problem: corrupt samples are written as separate events;
        # if a fixation/saccade is interrupted by a corrupt event, they are treated as separate
        if start_time:
            event_dat['start_time'] = pd.concat(
                [pd.Series([0]), event_dat.event_len.shift(1).cumsum().iloc[1:]])  # set first row of start_event to 0
            event_dat['end_time'] = event_dat.event_len.cumsum() - 1

        # if unit is pixel, add column with fixation means to events dataframe:
        if unit == 'px':
            means_x, means_y = extract_fixation_means(event_dat)
            event_dat['fix_mean_x'] = np.nan
            event_dat['fix_mean_y'] = np.nan
            event_dat.loc[event_dat.event == 1, 'fix_mean_x'] = means_x
            event_dat.loc[event_dat.event == 1, 'fix_mean_y'] = means_y

        if trial_based_time:
            print('TRIAL BASED TIME')
            # make time trial based:
            unique_trials = np.unique(event_dat.trial_id)
            for t in unique_trials:
                trial_start_time = event_dat.loc[event_dat.trial_id == t, 'start_time'].get_values()[0]
                event_dat.loc[event_dat.trial_id == t, 'start_time'] = event_dat.loc[
                    event_dat.trial_id == t, 'start_time',
                ] - trial_start_time
                event_dat.loc[event_dat.trial_id == t, 'end_time'] = event_dat.loc[
                    event_dat.trial_id == t, 'end_time',
                ] - trial_start_time

        # convert event length to durations in milliseconds
        event_dat['event_duration'] = event_dat['event_len'] * (1000 / Experiment_Class.sampling)

        event_dat = select_fixations(d, event_dat)
        event_dat = event_dat.drop(columns=['seq_t'])

        # function that returns fixations with word rois
        event_dat = get_aois_from_event_data(
            event_dat=event_dat,
            subj_dir=subj_dir,
            TRIAL_ID=TRIAL_ID,
            item_id=item_id,
            Trial_Index_=Trial_Index_,
        )

        filepath = create_filepath_for_event_file(
            subj_id=subj_id,
            event_dir=event_dir,
            item_id=item_id,
        )

        logging.info(f'writing to {filepath}')

        if plot_px_time:
            plot_px_over_time(
                subj_id=subj_id,
                event_dir=event_dir,
                sub_df=sub_df,
                event_dat=event_dat,
                TRIAL_ID=TRIAL_ID,
                item_id=item_id,
                Trial_Index_=Trial_Index_,
                model=model,
                decoding_strategy=decoding_strategy,
            )

        event_dat.to_csv(filepath, index=True, sep='\t', na_rep='NaN')


def select_fixations(d: pd.DataFrame, event_dat: pd.DataFrame):
    event_dat = event_dat[event_dat['event'] == 1]
    event_dat = event_dat.drop(columns=['seq_x', 'seq_y'])
    event_dat['index'] = range(1, len(event_dat) + 1)
    event_dat = event_dat.set_index('index')
    event_dat = event_dat.reindex(sorted(event_dat.columns), axis=1)
    return event_dat


def get_event_type(duration: float):
    if duration < 9:
        return 'microsaccade'
    else:
        return 'saccade'


# apply the function to create a new column called "type"
def select_saccades(d: pd.DataFrame, event_dat: pd.DataFrame):
    event_dat = event_dat[event_dat['event'] == 2]
    event_dat = event_dat.drop(columns=['seq_x', 'seq_y'])
    event_dat['index'] = range(1, len(event_dat) + 1)
    event_dat = event_dat.set_index('index')
    event_dat = event_dat.reindex(sorted(event_dat.columns), axis=1)
    event_dat['type'] = event_dat['event_duration'].apply(get_event_type)
    return event_dat


def choose_data_from_right(d: pd.DataFrame):
    d['x_deg'] = d['x_right_deg']
    d['y_deg'] = d['y_right_deg']
    d['dx_deg'] = d['dx_right_deg']
    d['dy_deg'] = d['dy_right_deg']
    d['x'] = d['x_right']
    d['y'] = d['y_right']
    d['t'] = d['time']


def choose_data_from_left(d: pd.DataFrame):
    d['x_deg'] = d['x_left_deg']
    d['y_deg'] = d['y_left_deg']
    d['dx_deg'] = d['dx_left_deg']
    d['dy_deg'] = d['dy_left_deg']
    d['x'] = d['x_left']
    d['y'] = d['y_left']
    d['t'] = d['time']


def continue_xy_event(row: pd.core.series.Series, unit: str, x: list[float], y: list[float], t: list[float]):
    if unit == 'velocity':
        x.append(row['dx_deg'])
        y.append(row['dy_deg'])
        t.append(row['t'])
    elif unit == 'deg':
        x.append(row['x_deg'])
        y.append(row['y_deg'])
        t.append(row['t'])
    else:
        x.append(row['x'])
        y.append(row['y'])
        t.append(row['t'])


def continue_xy_event_all(
        row: pd.core.series.Series,
        x_ddeg: float,
        y_ddeg: float,
        x_deg: float,
        y_deg: float,
        x_px: float,
        y_px: float,
        t: float,
):
    x_ddeg.append(row['dx_deg'])
    y_ddeg.append(row['dy_deg'])
    x_deg.append(row['x_deg'])
    y_deg.append(row['y_deg'])
    x_px.append(row['x'])
    y_px.append(row['y'])
    t.append(row['t'])


def update_xy_event(
        row: pd.core.series.Series,
        unit: str,
) -> tuple[list[float], list[float], list[float]]:
    if unit == 'velocity':
        x = [row['dx_deg']]
        y = [row['dy_deg']]
        t = [row['t']]
    elif unit == 'deg':
        x = [row['x_deg']]
        y = [row['y_deg']]
        t = [row['t']]
    else:
        x = [row['x']]
        y = [row['y']]
        t = [row['t']]
    return x, y, t


def update_xy_event_all(row: pd.core.series.Series) -> tuple[list[float], ...]:
    x_ddeg = [row['dx_deg']]
    y_ddeg = [row['dy_deg']]
    x_deg = [row['x_deg']]
    y_deg = [row['y_deg']]
    x_px = [row['x']]
    y_px = [row['y']]
    t = [row['t']]
    return x_ddeg, y_ddeg, x_deg, y_deg, x_px, y_px, t


def write_event_to_file(
        event: str,
        event_dat: pd.DataFrame,
        row: pd.core.series.Series,
        unit: str,
        x: float,
        y: float,
        t: float,
) -> pd.DataFrame:
    if unit == 'velocity':
        event_dat = pd.concat(
            [
                event_dat, pd.DataFrame(
                    {
                        'seq_dx_deg': [x],
                        'seq_dy_deg': [y],
                        'seq_t': [t],
                        'item_id': row['item_id'],
                        'TRIAL_ID': row['TRIAL_ID'],
                        'Trial_Index_': row['Trial_Index_'],
                        'subject_id': row['subject_id'],
                        'model': row['model'],
                        'decoding_strategy': row['decoding_strategy'],
                        'event': event,
                    },
                ),
            ], axis=0, sort=True,
        )
    elif unit == 'deg':
        event_dat = pd.concat(
            [
                event_dat, pd.DataFrame(
                    {
                        'seq_x_deg': [x],
                        'seq_y_deg': [y],
                        'seq_t': [t],
                        'item_id': row['item_id'],
                        'TRIAL_ID': row['TRIAL_ID'],
                        'Trial_Index_': row['Trial_Index_'],
                        'subject_id': row['subject_id'],
                        'model': row['model'],
                        'decoding_strategy': row['decoding_strategy'],
                        'event': event,
                    },
                ),
            ], axis=0, sort=True,
        )
    else:
        event_dat = pd.concat(
            [
                event_dat, pd.DataFrame(
                    {
                        'seq_x': [x],
                        'seq_y': [y],
                        'seq_t': [t],
                        'item_id': row['item_id'],
                        'TRIAL_ID': row['TRIAL_ID'],
                        'Trial_Index_': row['Trial_Index_'],
                        'subject_id': row['subject_id'],
                        'model': row['model'],
                        'decoding_strategy': row['decoding_strategy'],
                        'event': event,
                    },
                ),
            ], axis=0, sort=True,
        )
    return event_dat


def write_event_to_file_all(
        event: str,
        event_dat: pd.DataFrame,
        row: pd.core.series.Series,
        x_ddeg: float,
        y_ddeg: float,
        x_deg: float,
        y_deg: float,
        x_px: float,
        y_px: float,
        t: float,
) -> pd.DataFrame:
    event_dat = pd.concat(
        [
            event_dat, pd.DataFrame(
                {
                    'seq_dx_deg': [x_ddeg],
                    'seq_dy_deg': [y_ddeg],
                    'seq_x_deg': [x_deg],
                    'seq_y_deg': [y_deg],
                    'seq_x': [x_px],
                    'seq_y': [y_px],
                    'seq_t': [t],
                    'item_id': row['item_id'],
                    'TRIAL_ID': row['TRIAL_ID'],
                    'Trial_Index_': row['Trial_Index_'],
                    'subject_id': row['subject_id'],
                    'model': row['model'],
                    'decoding_strategy': row['decoding_strategy'],
                    'event': event,
                },
            ),
        ], axis=0, sort=True,
    )
    return event_dat


def main():
    start_time = datetime.now()

    args = get_parser().parse_args()

    config = load_config()
    exclude_subjects = config['exclude']['subjects']
    columns = config['csv2events']['columns']
    n_jobs = config['csv2events']['n_jobs']
    check_file_exists = config['csv2events']['check_file_exists']

    path_to_subjects = 'data/subject_level_data/'

    current_date = datetime.now()
    formatted_date = current_date.strftime('%Y-%m-%d')

    process_csv_to_events(
        path_to_subjects=path_to_subjects,
        columns=columns,
        exclude_subjects=exclude_subjects,
        n_jobs=n_jobs,
        threshold=args.threshold,
        threshold_factor=args.threshold_factor,
        threshold_method=args.threshold_method,
        min_fixation_duration_ms=args.min_fixation_duration_ms,
        min_saccade_duration_ms=args.min_saccade_duration_ms,
        max_saccade_velocity=args.max_saccade_velocity,
        theta=args.theta,
        check_file_exists=check_file_exists,
        disable_parallel=args.disable_parallel,
        plot_px_time=args.plot_px_time,
        plot_ampl_velocity=args.plot_ampl_vel,
    )

    logging.info(f'Took {datetime.now() - start_time} overall')


if __name__ == '__main__':
    raise SystemExit(main())
