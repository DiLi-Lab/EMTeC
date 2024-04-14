#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_events(events_df):
    trials = list(events_df.trialId.unique())

    fix_count_per_trial = []
    sacc_count_per_trial = []
    corrupt_count_per_trial = []
    for t in trials:
        fix_count_per_trial.append(np.sum(events_df.event[events_df.trialId==t] == 1))
        sacc_count_per_trial.append(np.sum(events_df.event[events_df.trialId == t] == 2))
        corrupt_count_per_trial.append(np.sum(events_df.event[events_df.trialId == t] == 3))

    plt.hist(fix_count_per_trial, alpha=0.5)
    plt.hist(sacc_count_per_trial, alpha=0.5)
    plt.hist(corrupt_count_per_trial, alpha=0.5)

    return 1


def plot_fixation_means(events_df, trialVars_df, trialId):
    events_df = events_df[events_df.trialId == trialId]

    assert len(events_df.trialId.unique()) == 1, 'More than one trial in event dataframe.'
    trialId = events_df.trialId.get_values()[0]

    plt.xlim([0, 1280])
    plt.ylim([0, 1024])

    # extract and plot means of the fixation events:
    if 'fix_mean_x' in events_df.columns and 'fix_mean_y' in events_df.columns:
        plt.scatter(events_df[events_df.event == 1].fix_mean_x, events_df[events_df.event == 1].fix_mean_y)
    else:
        means_x, means_y = extract_fixation_means(events_df)
        plt.scatter(means_x, means_y)

    # read and plot stimulus locations from trialVars:
    loc_x, loc_y = extract_stimulus_locations(trialVars_df, trialId)
    plt.scatter(loc_x, loc_y)
    return 1


def extract_fixation_means(events_df):
    means_x = []
    means_y = []

    for e in range(0, events_df.shape[0]):
        # assumes fixation events are coded as 1
        if events_df.event.values[e] == 1:
            means_x.append(np.mean(events_df.seq_x.values[e]))
            means_y.append(np.mean(events_df.seq_y.values[e]))

    return means_x, means_y


def extract_stimulus_locations(trialVars_df, trialId):
    # extract x and y coordinates of 5 jumping dots from trialVars dataframe

    # columns of locations
    cols_loc_x = ['locP' + str(i) + '_x' for i in [1, 2, 3, 4, 5]]
    cols_loc_y = ['locP' + str(i) + '_y' for i in [1, 2, 3, 4, 5]]

    loc_x = trialVars_df[trialVars_df.trialId == trialId][cols_loc_x].get_values()
    loc_y = trialVars_df[trialVars_df.trialId == trialId][cols_loc_y].get_values()

    return loc_x, loc_y