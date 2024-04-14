#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions for saccade detection etc.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def pix2deg(pix, screenPX, screenCM, distanceCM, adjust_origin=True):
    """
    Converts pixel screen coordinate to degrees of visual angle
    screenPX is the number of pixels that the monitor has in the horizontal axis (for x coord) or vertical axis
    (for y coord)
    screenCM is the width of the monitor in centimeters
    distanceCM is the distance of the monitor to the retina
    pix: screen coordinate in pixels
    adjust origin: if origin (0,0) of screen coordinates is in the corner of the screen rather than in the center, set to True to center coordinates
    """
    pix = np.array(pix)
    # center screen coordinates such that 0 is center of the screen:
    if adjust_origin:
        pix = pix - (screenPX - 1) / 2  # pixel coordinates start with (0,0)
    # eye-to-screen-distance in pixels
    distancePX = distanceCM * (screenPX / screenCM)
    return (
        np.arctan2(pix, distancePX) * 180 / np.pi
    )  # x180/pi wandelt bogenmass in grad


def microsacc(
        velocities: np.array,
        threshold: Tuple[float, float],
        threshold_factor: float,
        min_duration: int,
        sampling_rate: int,
):
    """
    Detect microsaccades in eye movement velocity data.

    Parameters
    ----------
    velocities : array-like, shape (N,)
        Eye velocity data.
    threshold_factor : float, optional (default=6)
        A factor used to determine the velocity threshold for microsaccade detection.
        The threshold is calculated as `threshold_factor` times the median absolute deviation of the velocities.
    min_duration : int, optional (default=6)
        The minimum duration of a microsaccade, in number of samples.
    sampling_rate : int, optional (default=1000)
        The sampling rate of the eye movement data, in Hz.
    threshold : float, optional (default=None)
        A specific velocity threshold to use for microsaccade detection.
        If `threshold` is provided, `threshold_factor` will be ignored.

    Returns
    -------
    sac : array, shape (8,)
        An array containing the following information for each detected microsaccade:
        - Saccade onset (sample nr.)
        - Saccade offset (sample nr.)
        - Peak velocity
        - Horizontal component (distance from first to last sample of the saccade)
        - Vertical component (distance from first to last sample of the saccade)
        - Horizontal amplitude (distance from leftmost to rightmost sample of the saccade)
        - Vertical amplitude (distance from topmost to bottommost sample of the saccade)
    issac : array, shape (N,)
        A binary array indicating whether each velocity value corresponds to a microsaccade (1) or not (0).
    radius : tuple, shape (2,)
        A tuple containing the horizontal and vertical semi-axes of the elliptic threshold used for microsaccade detection.
    """
    v = vecvel(velocities, sampling_rate=sampling_rate)
    # if threshold:  # global threshold provided
    #     msdx, msdy = threshold[0], threshold[1]
    # else:
    #     msdx, msdy = estimate_threshold(v)
    msdx, msdy = threshold

    radiusx = threshold_factor * msdx  # x-radius of elliptic threshold
    radiusy = threshold_factor * msdy  # y-radius of elliptic threshold
    radius = (radiusx, radiusy)
    # test if sample is within elliptic threshold

    test = np.power((v[:, 0] / radiusx), 2) + np.power(
        (v[:, 1] / radiusy), 2
    )  # test is <1 iff sample within ellipse
    indx = np.where(np.greater(test, 1))[
        0
    ]  # indices of candidate saccades; runtime warning because of nans in test => is ok, the nans come from nans in x
    # Determine saccades
    N = len(indx)  # anzahl der candidate saccades
    nsac = 0
    sac = []
    dur = 1
    a = 0  # (möglicher) begin einer saccade
    k = 0  # (möglisches) ende einer saccade, hierueber wird geloopt
    issac = np.zeros(len(velocities))  # codes if x[issac] is a saccade

    # Loop over saccade candidates
    while k < N - 1:  # loop over candidate saccades
        if indx[k + 1] - indx[k] == 1:  # saccade candidates, die aufeinanderfolgen
            dur += 1  # erhoehe sac dur
        else:  # wenn nicht (mehr) in saccade
            # Minimum duration criterion (exception: last saccade)
            if dur >= min_duration:  # schreibe saccade, sofern MINDUR erreicht wurde
                nsac += 1
                s = np.zeros(8)  # entry for this saccade
                s[0] = indx[a]  # saccade onset
                s[1] = indx[k]  # saccade offset
                sac.append(s)
                issac[indx[a] : indx[k] + 1] = 1  # code as saccade from onset to offset
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
        issac[indx[a] : indx[k] + 1] = 1  # code as saccade from onset to offset
    sac = np.array(sac)
    # print("number of saccades detected: ", nsac)
    if nsac > 0:
        # Compute peak velocity, horiztonal and vertical components
        for s in range(nsac):  # loop over saccades
            # Onset and offset for saccades
            a = int(sac[s, 0])  # onset of saccade s
            b = int(sac[s, 1])  # ofefset of saccade s
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
            sac[s, 7] = b - a  # duration??
    return sac, issac, radius


# def estimate_threshold(v):
#     # Compute threshold from data (as Engbert et al)
#     msdx = np.sqrt(
#         np.nanmedian(np.power(v[:, 0] - np.nanmedian(v[:, 0]), 2))
#     )  # median-based std of x-velocity
#     msdy = np.sqrt(np.nanmedian(np.power(v[:, 1] - np.nanmedian(v[:, 1]), 2)))
#     assert msdx > 1e-10  # there must be enough variance in the data
#     assert msdy > 1e-10
#     return msdx, msdy


def estimate_threshold(
        velocities: np.array,
        threshold_method: str,
):
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

    # TODO maybe comment in? it's in Lena's original implementation but not in the pymovements implementation
    # # there must be enough variance in the data
    # assert thx > 1e-10
    # assert thy > 1e-10

    return thx, thy


def issac(
    velocities,
    threshold_factor=6,
    min_duration=6,
    sampling_rate=1000,
    threshold=(10, 10),
):  # median/mean threshold von allen texten: (9,6)
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
                is_sac[
                    indx[a] : indx[k] + 1
                ] = 1  # code as saccade from onset to offset
            a = k + 1  # potential onset of next saccade
            dur = 1  # reset duration
        k = k + 1
    # Check minimum duration for last microsaccade
    if dur >= min_duration:
        is_sac[indx[a] : indx[k] + 1] = 1  # code as saccade from onset to offset
    return is_sac


def corruptSamplesIdx(x, y, x_max, y_max, x_min, y_min, theta=0.6, samplingRate=1000):
    """
    corruptSamples nach Sakkadenerkennung anwenden, damit Chronologie erhalten bleibt
    es kann nach Anwendung von corruptSamples Fixationen und Sakkaden der Laenge 1 geben.
    x,y: x,y coordinates in degrees of visual anngle
    x,y must be chronologically ordered sample sequences, missing values have NOT been removed yet
    max_x... max/min coordinates for samples to fall on the screen;
    theta: velocity threshold in degrees/ms
    """
    x = np.array(x)
    y = np.array(y)
    # Find Offending Samples
    # samples that exceed velocity threshold
    # adjust theta depending on sampling rate
    theta = theta * 1000 / samplingRate
    x2 = np.append(
        x[0], x[:-1]
    )  # x2 is sample that precedes x (numpy.roll not used here beacuse it inserts last element in first position)
    y2 = np.append(y[0], y[:-1])
    distTrv = np.sqrt(np.power((x - x2), 2) + np.power((y - y2), 2))
    fast_ix = np.where(distTrv > theta)[0]  # too fast samples; np.where returns tuple
    # Missing samples
    mis_ix = np.where(np.isnan(x) | np.isnan(y))[0]  # np.where returns tuple of len 1
    # samples outside the screen
    out_ix = np.where(
        np.greater(x, x_max)
        | np.greater(y, y_max)
        | np.greater(x_min, x)
        | np.greater(y_min, y)
    )[0]
    # RuntimeWarning: invalid value encountered in greater => is ok; comparison with nan yields nan
    # remove samples,where gaze is completely still => scheint auch recording Fehler zu sein
    # TODO: macht das Sinn? Hiervon sind meist nur einzelne samples betroffen. erstmal raus. ggf wieder einkommentieren.
    # still_ix = np.where(np.equal(x,np.roll(x, shift=-1)) & np.equal(x,np.roll(x, shift=1)) & np.equal(y,np.roll(y, shift=-1)) & np.equal(y,np.roll(y, shift=1)))[0]
    # return all bad samples
    return np.sort(np.unique(np.concatenate((mis_ix, out_ix, fast_ix))))


def vecvel(x, sampling_rate=1000, smooth=True):
    """
    Compute velocity times series from 2D position data
    adapted from Engbert et al.  Microsaccade Toolbox 0.9
    x: array of shape (N,2) (x und y screen or visual angle coordinates of N samples in *chronological* order)
    returns velocity in deg/sec or pix/sec
    """
    N = x.shape[0]
    v = np.zeros((N, 2))  # first column for x-velocity, second column for y-velocity
    if smooth:  # v based on mean of preceding 2 samples and mean of following 2 samples
        v[2 : N - 2, :] = (sampling_rate / 6) * (
            x[4:N, :] + x[3 : N - 1, :] - x[1 : N - 3, :] - x[0 : N - 4, :]
        )
        # *SAMPLING => pixeldifferenz pro sec
        # v[n,:]: Differenz zwischen mean(sample_n-2, sample_n-1) und mean(sample_n+1, sample_n+2) (=> durch 2 teilen); jetzt ist aber Schrittweite 3 sample lang (von n-1.5 bis n+1.5) => durch 3 teilen => insgesamt durch 6 teilen
        # v based on preceding sample and following sample for second and penultimate sample
        v[1, :] = (sampling_rate / 2) * (x[2, :] - x[0, :])
        v[N - 2, :] = (sampling_rate / 2) * (x[N - 1, :] - x[N - 3, :])
    else:
        v[1 : N - 1,] = (sampling_rate / 2) * (
            x[2:N, :] - x[0 : N - 2, :]
        )  # differenz Vorgänger sample und nachfolger sample; beginnend mit 2.sample bis N
    # /2 weil dx differenz zwischen voergänger und nachfolger sample ist => schrittweite ist also 2
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

