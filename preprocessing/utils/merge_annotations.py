#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to merge the word-level annotations with the reading measures files.
"""

import numpy as np
import pandas as pd


def main():

    # read in the files

    path_to_word_level_annotations = 'annotation/word_level_annotations.csv'
    path_to_rms = 'data/reading_measures.csv'
    path_to_rms_corrected = 'data/reading_measures_corrected.csv'
    path_to_stimuli = 'data/stimuli.csv'

    annotations = pd.read_csv(path_to_word_level_annotations, sep='\t')
    rms = pd.read_csv(path_to_rms, sep='\t')
    rms_corrected = pd.read_csv(path_to_rms_corrected, sep='\t')
    stimuli = pd.read_csv(path_to_stimuli, sep='\t')

    # merge the annotations with the reading measures
    print(' --- merging annotations with reading measures')
    merged = pd.merge(rms, annotations, on=['item_id', 'model', 'decoding_strategy', 'word_id'], how='left')
    print(' --- merging corrected annotations with corrected reading measures')
    merged_corrected = pd.merge(rms_corrected, annotations,
                                on=['item_id', 'model', 'decoding_strategy', 'word_id'], how='left')

    # merge the reading measures with the stimulus info type/task/subcategory
    print(' --- merging reading measures with stimulus info')
    merged = pd.merge(merged, stimuli[['item_id', 'model', 'decoding_strategy', 'type', 'task', 'subcategory']],
                      on=['item_id', 'model', 'decoding_strategy'])
    print(' --- merging corrected reading measures with stimulus info')
    merged_corrected = pd.merge(merged_corrected,
                                stimuli[['item_id', 'model', 'decoding_strategy', 'type', 'task', 'subcategory']],
                                on=['item_id', 'model', 'decoding_strategy'])

    # save the merged files
    merged.to_csv(path_to_rms, sep='\t', index=False)
    merged_corrected.to_csv(path_to_rms_corrected, sep='\t', index=False)


if __name__ == '__main__':
    raise SystemExit(main())
