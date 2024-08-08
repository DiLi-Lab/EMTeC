#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd


def extract_fixation_means(events_df: pd.DataFrame) -> tuple[list[float], ...]:
    """
    Extracts the mean x and y coordinates of fixations from the events dataframe.
    :param events_df: pandas dataframe with columns 'seq_x', 'seq_y', 'event'
    :return: list of mean x coordinates, list of mean y coordinates
    """
    means_x = []
    means_y = []

    for e in range(0, events_df.shape[0]):
        # assumes fixation events are coded as 1
        if events_df.event.values[e] == 1:
            means_x.append(np.mean(events_df.seq_x.values[e]))
            means_y.append(np.mean(events_df.seq_y.values[e]))

    return means_x, means_y
