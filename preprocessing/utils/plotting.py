#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions to plot the tracked coordinates and saccade velocity over amplitude.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List
from preprocessing.parsing.ed_helpers import compute_peak_velocity_and_amplitude


def plot_px_over_time(
    subj_id: str,
    event_dir: str,
    sub_df: pd.DataFrame,
    event_dat: pd.DataFrame,
    TRIAL_ID: int,
    item_id: str,
    Trial_Index_: int,
    model: str,
    decoding_strategy: str,
):
    """
    Function to plot the raw x- and y- coordinates over time and insert the fixations extracted by the algorith into
    the plot.
    """
    plot_filename = f'{subj_id}_trialid{TRIAL_ID}_{item_id}_trialindex{Trial_Index_}_{model}_{decoding_strategy}_px_time.png'
    plot_savepath = os.path.join(event_dir, 'plots', 'px_time')
    if not os.path.exists(plot_savepath):
        os.makedirs(plot_savepath)

    xs = sub_df['x_right'].tolist()
    ys = sub_df['y_right'].tolist()
    time = sub_df['time'].tolist()
    fix_intervals = [
        (t_start, t_end) for t_start, t_end in zip(event_dat['t_start'].tolist(), event_dat['t_end'].tolist())
    ]
    fig, ax = plt.subplots(figsize=(24, 6))
    # ax.plot(time, xs, 'c', label='horizontal movement')
    # ax.plot(time, ys, 'm', label='vertical movement')
    ax.plot(time, xs, color='darkcyan', label='horizontal movement')
    ax.plot(time, ys, color='mediumorchid', label='vertical movement')
    for index, interval in enumerate(fix_intervals):
        if index == 0:
            plt.axvspan(interval[0], interval[1], facecolor='grey', alpha=.3, label='Fixation')
        else:
            plt.axvspan(interval[0], interval[1], facecolor='grey', alpha=.3)
    ax.set_xlabel('time')
    ax.set_ylabel('position in pixels')
    ax.legend(loc='upper left')
    plt.title(f'Coordinates over time with extracted fixations')
    plt.savefig(os.path.join(plot_savepath, plot_filename))
    plt.close()


# def plot_ampl_over_vel(
#     subj_id: str,
#     event_dir: str,
#     saccade_info,
#     sampling: int,
#     TRIAL_ID: int,
#     item_id: str,
#     Trial_Index_: int,
#     model: str,
#     decoding_strategy: str,
# ):
#     plot_filename = f'{subj_id}_trialid{TRIAL_ID}_{item_id}_trialindex{Trial_Index_}_{model}_{decoding_strategy}_ampl_vel.png'
#     plot_savepath = os.path.join(event_dir, 'plots', 'ampl_vel')
#     if not os.path.exists(plot_savepath):
#         os.makedirs(plot_savepath)
#
#     duration = saccade_info[:, 1] - saccade_info[:, 0]
#     duration = duration / (sampling / 1000)  # sampling freq 2000: if we look at e.g. 40 samples, divide by 2 to get 20ms
#     amplitude = np.sqrt(saccade_info[:, 5] ** 2 + saccade_info[:, 6] ** 2)
#     peak_velocity = saccade_info[:, 2]
#
#     # plot the main sequence
#     plt.plot(amplitude, peak_velocity, 'o')
#     plt.xlabel('amplitude')
#     plt.ylabel('peak velocity')
#
#     plt.savefig(os.path.join(plot_savepath, plot_filename))
#     plt.close()


def plot_ampl_over_vel(
        event_dat: pd.DataFrame,
        sampling: int,
        subj_id: str,
        event_dir: str,
        TRIAL_ID: int,
        item_id: str,
        Trial_Index_: int,
        model: str,
        decoding_strategy: str,
):
    plot_filename = f'{subj_id}_trialid{TRIAL_ID}_{item_id}_trialindex{Trial_Index_}_{model}_{decoding_strategy}_ampl_vel_regline.png'
    plot_savepath = os.path.join(event_dir, 'plots', 'ampl_vel_processed')
    if not os.path.exists(plot_savepath):
        os.makedirs(plot_savepath)
    # peak_velocities, amplitudes = compute_peak_velocity_and_amplitude(event_dat=event_dat, sampling=sampling)

    amplitudes = event_dat['amplitude'].tolist()
    peak_velocities = event_dat['peak_velocity'].tolist()
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(amplitudes, peak_velocities)
    # plot peak velocity over amplitude
    plt.plot(amplitudes, peak_velocities, 'o', color='steelblue', markersize=3, label='Data points')
    # plot regression line
    plt.plot(amplitudes, intercept + slope * np.array(amplitudes), color='salmon', label='Regression Line')
    # # Add labels for each point
    # for i, txt in enumerate(range(1, len(amplitudes) + 1)):
    #     plt.annotate(txt, (amplitudes[i], peak_velocities[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.xlabel('amplitude deg')
    plt.ylabel('peak velocity deg/s')

    # Set y-axis limits
    plt.ylim(0, max(peak_velocities)+100)

    plt.savefig(os.path.join(plot_savepath, plot_filename))
    plt.close()

    # plot once without the regression line
    plot_filename = f'{subj_id}_trialid{TRIAL_ID}_{item_id}_trialindex{Trial_Index_}_{model}_{decoding_strategy}_ampl_vel.png'
    # plot peak velocity over amplitude
    plt.plot(amplitudes, peak_velocities, 'o', color='steelblue', markersize=3, label='Data points')
    plt.xlabel('amplitude deg')
    plt.ylabel('peak velocity deg/s')

    # Set y-axis limits
    plt.ylim(0, max(peak_velocities) + 100)

    plt.savefig(os.path.join(plot_savepath, plot_filename))
    plt.close()


def plot_ampl_over_vel_all_trials(
        velocities: List[List[float]],
        amplitudes: List[List[float]],
        subj_id: str,
        event_dir: str,
):
    plot_filename = f'{subj_id}_ampl_vel_all_trials.png'
    plot_savepath = os.path.join(event_dir, 'plots', 'ampl_vel_processed')
    if not os.path.exists(plot_savepath):
        os.makedirs(plot_savepath)
    # perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(amplitudes, velocities)
    # plot peak velocity over amplitude
    plt.plot(amplitudes, velocities, 'o', color='steelblue')
    # plot regression line
    plt.plot(amplitudes, intercept + slope * np.array(amplitudes), color='salmon')

    plt.xlabel('amplitude deg')
    plt.ylabel('peak velocity deg/s')

    # Set y-axis limits
    plt.ylim(0, max(velocities) + 100)

    plt.savefig(os.path.join(plot_savepath, plot_filename))
    plt.close()