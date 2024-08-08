#!/usr/bin/env python3
"""
Functions for saccade detection.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def pix2deg(
        pix: Sequence,
        screenPX: Sequence,
        screenCM: Sequence,
        distanceCM: Sequence,
        adjust_origin: bool = True,
):
    """
    Converts pixel screen coordinate to degrees of visual angle
    :param pix: the screen coordinate in pixels
    :param screenPX: the number of pixels that the monitor has in the horizontal axis (for x coord) or vertical axis
    :param screenCM: the width of the monitor in centimeters
    :param distanceCM: the distance of the monitor to the retina
    :param adjust_origin: if origin (0,0) of screen coordinates is in the corner of the screen rather than in the center,
    """
    pix = np.array(pix)
    # center screen coordinates such that 0 is center of the screen:
    if adjust_origin:
        pix = pix - (screenPX - 1) / 2  # pixel coordinates start with (0,0)
    # eye-to-screen-distance in pixels
    distancePX = distanceCM * (screenPX / screenCM)
    return (
        np.arctan2(pix, distancePX) * 180 / np.pi
    )


def microsacc(
        velocities: np.array,
        threshold: tuple[float, float],
        threshold_factor: float,
        min_duration: int,
        sampling_rate: int,
):
    """
    Detect microsaccades in eye movement velocity data.
    :param velocities: Eye velocity data. array of shape (N,)
    :param threshold: A specific velocity threshold to use for microsaccade detection. If 'threshold' is provided,
    'threshold_factor' will be ignored. Tuple of shape (2,)
    :param threshold_factor: A factor used to determine the velocity threshold for microsaccade detection. The threshold
    is calculated as 'threshold_factor' times the median absolute deviation of the velocities. float
    :param min_duration: The minimum duration of a microsaccade, in number of samples. int
    :param sampling_rate: The sampling rate of the eye movement data, in Hz. int
    :return: A tuple containing the following:
    - sac: An array containing the following information for each detected microsaccade:
        - Saccade onset (sample nr.)
        - Saccade offset (sample nr.)
        - Peak velocity
        - Horizontal component (distance from first to last sample of the saccade)
        - Vertical component (distance from first to last sample of the saccade)
        - Horizontal amplitude (distance from leftmost to rightmost sample of the saccade)
        - Vertical amplitude (distance from topmost to bottommost sample of the saccade)
    - issac: A binary array indicating whether each velocity value corresponds to a microsaccade (1) or not (0).
    - radius: A tuple containing the horizontal and vertical semi-axes of the elliptic threshold used for microsaccade
    """
    # compute velocity time series from 2D position data
    v = vecvel(velocities, sampling_rate=sampling_rate)
    msdx, msdy = threshold

    radiusx = threshold_factor * msdx  # x-radius of elliptic threshold
    radiusy = threshold_factor * msdy  # y-radius of elliptic threshold
    radius = (radiusx, radiusy)

    # test if sample is within elliptic threshold
    test = np.power((v[:, 0] / radiusx), 2) + np.power(
        (v[:, 1] / radiusy), 2
    )  # test is <1 iff sample within ellipse

    # indices of candidate saccades; runtime warning because of nans in test => is ok, the nans come from nans in x
    indx = np.where(np.greater(test, 1))[0]

    # Determine saccades
    N = len(indx)  # number of candidate saccades
    nsac = 0
    sac = []
    dur = 1
    a = 0  # (possible) onset of a saccade
    k = 0  # (possible) offset of a saccade, loop over this
    issac = np.zeros(len(velocities))  # codes if x[issac] is a saccade

    # Loop over saccade candidates
    while k < N - 1:  # loop over candidate saccades
        # if we are within one saccade
        if indx[k + 1] - indx[k] == 1:  # consecutive saccade candidates
            dur += 1  # increment saccade durations
        # else if we are not in a saccade anymore
        else:
            # Minimum duration criterion (exception: last saccade)
            if dur >= min_duration:  # write saccade if it fulfils its minimum duration criterion
                nsac += 1
                s = np.zeros(8)  # entry for this saccade
                s[0] = indx[a]  # saccade onset
                s[1] = indx[k]  # saccade offset
                sac.append(s)
                issac[indx[a]:indx[k] + 1] = 1  # code as saccade from onset to offset
            a = k + 1  # potential onset of next saccade
            dur = 1  # reset duration
        k = k + 1

    # Check minimum duration for last microsaccade
    if dur >= min_duration:
        nsac += 1
        s = np.zeros(8)  # entry for this saccade
        s[0] = indx[a]  # saccade onset
        s[1] = indx[k]  # saccade offset
        sac.append(s)
        issac[indx[a]:indx[k] + 1] = 1  # code as saccade from onset to offset
    sac = np.array(sac)

    if nsac > 0:
        # Compute peak velocity, horiztonal and vertical components
        for s in range(nsac):  # loop over saccades

            # Onset and offset for saccades
            a = int(sac[s, 0])  # onset of saccade s
            b = int(sac[s, 1])  # offset of saccade s
            idx = range(a, b + 1)  # indices of samples belonging to saccade s

            # Saccade peak velocity (vpeak)
            sac[s, 2] = np.max(np.sqrt(np.power(v[idx, 0], 2) + np.power(v[idx, 1], 2)))

            # saccade length measured as distance between first (onset) and last (offset) sample
            sac[s, 3] = velocities[b, 0] - velocities[a, 0]
            sac[s, 4] = velocities[b, 1] - velocities[a, 1]

            # Saccade amplitude: saccade length measured as distance between leftmost and rightmost (bzw. highest and lowest) sample
            minx = np.min(velocities[idx, 0])  # smallest x-coordinate during saccade
            maxx = np.max(velocities[idx, 0])
            miny = np.min(velocities[idx, 1])
            maxy = np.max(velocities[idx, 1])
            signx = np.sign(
                np.where(velocities[idx, 0] == maxx)[0][0]
                - np.where(velocities[idx, 0] == minx)[0][0]
            )  # direction of saccade; np.where returns tuple; there could be more than one minimum/maximum => chose the first one
            signy = np.sign(
                np.where(velocities[idx, 1] == maxy)[0][0]
                - np.where(velocities[idx, 1] == miny)[0][0]
            )  #
            sac[s, 5] = signx * (maxx - minx)  # x-amplitude
            sac[s, 6] = signy * (maxy - miny)  # y-amplitude
            sac[s, 7] = b - a  # duration of saccade

    return sac, issac, radius


def estimate_threshold(
        velocities: np.array,
        threshold_method: str,
) -> Tuple[float, float]:
    """
    Estimate the velocity threshold for microsaccade detection.
    :param velocities: Eye velocity data. array of shape (N,)
    :param threshold_method: The method used to estimate the threshold. str
    :return: A tuple containing the horizontal and vertical semi-axes of the elliptic threshold used for microsaccade
    """
    if threshold_method == 'std':
        thx = np.nanstd(velocities[:, 0])
        thy = np.nanstd(velocities[:, 1])
    elif threshold_method == 'mad':
        thx = np.nanmedian(np.absolute(velocities[:, 0] - np.nanmedian(velocities[:, 0])))
        thy = np.nanmedian(np.absolute(velocities[:, 1] - np.nanmedian(velocities[:, 1])))
    elif threshold_method == 'engbert2003':
        thx = np.sqrt(
            np.nanmedian(np.power(velocities[:, 0], 2)) - np.power(np.nanmedian(velocities[:, 0]), 2),
        )
        thy = np.sqrt(
            np.nanmedian(np.power(velocities[:, 1], 2)) - np.power(np.nanmedian(velocities[:, 1]), 2),
        )
    elif threshold_method == 'engbert2015':
        thx = np.sqrt(
            np.nanmedian(np.power(velocities[:, 0] - np.nanmedian(velocities[:, 0]), 2))
        )
        thy = np.sqrt(
            np.nanmedian(np.power(velocities[:, 1] - np.nanmedian(velocities[:, 1]), 2))
        )
    else:
        valid_methods = ['std', 'mad', 'engbert2003', 'engbert2015']
        raise ValueError(f'Method {threshold_method} not implemented. Valid methods: {valid_methods}')

    return thx, thy


def issac(
        velocities: np.ndarray,
        threshold_factor: float | int = 6,
        min_duration: float | int = 6,
        sampling_rate: float | int = 1000,
        threshold: tuple[int, int] = (10, 10),
):
    """
    Detect whether each velocity value corresponds to a microsaccade.
    :param velocities: Eye velocity data. array of shape (N,)
    :param threshold_factor: A factor used to determine the velocity threshold for microsaccade detection. The threshold
    is calculated as 'threshold_factor' times the median absolute deviation of the velocities. float
    :param min_duration: The minimum duration of a microsaccade, in number of samples. int
    :param sampling_rate: The sampling rate of the eye movement data, in Hz. int
    :param threshold: A specific velocity threshold to use for microsaccade detection. If 'threshold' is provided,
    'threshold_factor' will be ignored. Tuple of shape (2,)
    :return: A binary array indicating whether each velocity value corresponds to a microsaccade (1) or not (0).
    """
    # median/mean threshold von allen texten: (9,6)
    # Compute velocity
    v = vecvel(velocities, sampling_rate=sampling_rate)
    if threshold:  # global threshold provided
        msdx, msdy = threshold[0], threshold[1]
    else:
        # Compute threshold
        msdx = np.sqrt(
            np.nanmedian(np.power(v[:, 0] - np.nanmedian(v[:, 0]), 2))
        )  # median-based std of x-velocity
        msdy = np.sqrt(np.nanmedian(np.power(v[:, 1] - np.nanmedian(v[:, 1]), 2)))
        assert msdx > 1e-10  # there must be enough variance in the data
        assert msdy > 1e-10

    radiusx = threshold_factor * msdx  # x-radius of elliptic threshold
    radiusy = threshold_factor * msdy  # y-radius of elliptic threshold
    # test if sample is within elliptic threshold
    test = np.power((v[:, 0] / radiusx), 2) + np.power(
        (v[:, 1] / radiusy), 2
    )  # test is <1 iff sample within ellipse
    indx = np.where(np.greater(test, 1))[
        0
    ]  # indices of candidate saccades; runtime warning because of nans in test => is ok, the nans come from nans in x
    # Determine saccades
    N = len(indx)  # anzahl der candidate saccades
    dur = 1
    a = 0  # (möglicher) begin einer saccade
    k = 0  # (möglisches) ende einer saccade, hierueber wird geloopt
    is_sac = np.zeros(len(velocities))  # codes if x[issac] is a saccade
    # Loop over saccade candidates
    while k < N - 1:  # loop over candidate saccades
        if indx[k + 1] - indx[k] == 1:  # saccade candidates, die aufeinanderfolgen
            dur = dur + 1  # erhoehe sac dur
        else:  # wenn nicht (mehr) in saccade
            # Minimum duration criterion (exception: last saccade)
            if dur >= min_duration:  # schreibe saccade, sofern MINDUR erreicht wurde
                is_sac[indx[a]:indx[k] + 1] = 1  # code as saccade from onset to offset
            a = k + 1  # potential onset of next saccade
            dur = 1  # reset duration
        k = k + 1
    # Check minimum duration for last microsaccade
    if dur >= min_duration:
        is_sac[indx[a]:indx[k] + 1] = 1  # code as saccade from onset to offset
    return is_sac


def corruptSamplesIdx(
        x: Sequence,
        y: Seqeunce,
        x_max: float | int,
        y_max: float | int,
        x_min: float | int,
        y_min: float | int,
        theta: float = 0.6,
        samplingRate: float | int = 1000,
):
    """
    Find samples that are corrupt (e.g. due to blinks) based on velocity threshold.
    After applying this function, there might be saccaes of length 1.
    :param x: x coordinates in degrees of visual angle
    :param y: y coordinates in degrees of visual angle
    :param x_max: maximum x coordinate for samples to fall on the screen
    :param y_max: maximum y coordinate for samples to fall on the screen
    :param x_min: minimum x coordinate for samples to fall on the screen
    :param y_min: minimum y coordinate for samples to fall on the screen
    :param theta: velocity threshold in degrees/ms
    :param samplingRate: the sampling rate of the eye movement data, in Hz
    :return: An array containing the indices of the corrupt samples
    """
    x = np.array(x)
    y = np.array(y)

    # find offending samples that exceed velocity threshold
    # adjust theta depending on sampling rate
    theta = theta * 1000 / samplingRate

    # x2 and y2 are the samples that precede x and y
    x2 = np.append(x[0], x[:-1])  # x2 = np.roll(x, shift=1)
    y2 = np.append(y[0], y[:-1])

    distTrv = np.sqrt(np.power((x - x2), 2) + np.power((y - y2), 2))

    # too fast samples, np.where returns tuple
    fast_ix = np.where(distTrv > theta)[0]
    # Missing samples
    mis_ix = np.where(np.isnan(x) | np.isnan(y))[0]  # np.where returns tuple of len 1
    # samples outside the screen
    out_ix = np.where(
        np.greater(x, x_max)
        | np.greater(y, y_max)
        | np.greater(x_min, x)
        | np.greater(y_min, y),
    )[0]

    return np.sort(np.unique(np.concatenate((mis_ix, out_ix, fast_ix))))


def vecvel(
        x: np.ndarray,
        sampling_rate: float | int = 1000,
        smooth: bool = True,
):
    """
    Compute velocity times series from 2D position data
    adapted from Engbert et al. 2015, Microsaccade Toolbox 0.9
    :param x: array of shape (N,2) (x and y screen or visual angle coordinates of N samples in *chronological* order)
    :param sampling_rate: the sampling rate of the eye movement data, in Hz
    :param smooth: whether to smooth the velocity time series
    :return: velocity in deg/sec or pix/sec
    """

    N = x.shape[0]
    # first column for x-velocity, second column for y-velocity
    v = np.zeros((N, 2))

    # v based on preceding sample and following sample for first and last sample
    if smooth:
        v[2 : N - 2, :] = (sampling_rate / 6) * (
            x[4:N, :] + x[3 : N - 1, :] - x[1 : N - 3, :] - x[0 : N - 4, :]
        )
        # v based on preceding sample and following sample for second and penultimate sample
        # divide by 6 because the step size is 3
        v[1, :] = (sampling_rate / 2) * (x[2, :] - x[0, :])
        v[N - 2, :] = (sampling_rate / 2) * (x[N - 1, :] - x[N - 3, :])
    else:
        # difference between preceding and following sample; start with 2. sample until N
        # divide by 2 because dx is the difference between the preceding and following sample => step size is 2
        v[1 : N - 1,] = (sampling_rate / 2) * (
            x[2:N, :] - x[0 : N - 2, :]
        )

    return v


def compute_peak_velocity_and_amplitude(
        event_dat: pd.DataFrame,
        sampling: int,
):
    """
    Compute the peak velocity in degrees/ms, and saccade amplitude
     :param event_dat: a dataframe containing one row for each event (saccade, fixation, or corrupt) and an array of
     degrees of visual angle for x and y for each event.
     :param sampling: the sampling frequency
    """
    # each row has index 0 in the event_dat DF; reset
    event_dat.reset_index(drop=True, inplace=True)

    # peak velocity for each saccade
    peak_velocities = list()
    # amplitude for each saccade
    amplitudes = list()

    for row_idx, row in event_dat.iterrows():

        # if fixation (1) or corrupt (3): continue; saccades are coded as 2
        if row['event'] != 2:
            # fixations and corrupt events don't have peak velocities or amplitudes
            peak_velocities.append(-1)
            amplitudes.append(-1)
            continue

        # peak velocity
        deg_seqs = np.transpose(np.array([row['seq_x_deg'], row['seq_y_deg']]))
        try:
            velocities = vecvel(x=deg_seqs, sampling_rate=sampling)
        except IndexError:
            breakpoint()
        peak_vel = np.max(np.sqrt(np.power(velocities[:, 0], 2) + np.power(velocities[:, 1], 2)))
        peak_velocities.append(peak_vel)

        # saccade amplitude
        minx = np.min(deg_seqs[:, 0])  # smallest x-coordinate during saccade
        maxx = np.max(deg_seqs[:, 0])
        miny = np.min(deg_seqs[:, 1])
        maxy = np.min(deg_seqs[:, 1])
        # direction of saccade: np.where returns tuple: there could be more than one minimum/maximum. choose the first one.
        signx = np.sign(
            np.where(deg_seqs[:, 0] == maxx)[0][0] - np.where(deg_seqs[:, 0] == minx)[0][0]
        )
        signy = np.sign(
            np.where(deg_seqs[:, 1] == maxy)[0][0] - np.where(deg_seqs[:, 1] == miny)[0][0]
        )
        x_ampl = signx * (maxx - minx)
        y_ampl = signy * (maxy - miny)
        ampl = np.sqrt(np.power(x_ampl, 2) + np.power(y_ampl, 2))
        amplitudes.append(ampl)

    return peak_velocities, amplitudes
