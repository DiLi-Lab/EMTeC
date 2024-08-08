import pandas as pd
import numpy as np
from annotation.readability_local.readability import Readability
import os
from wordfreq import word_frequency, zipf_frequency

def main():

    text_lengths = []  # number of words in text
    avg_word_freqs = []  # average word frequency
    avg_zipf_freqs = []  # average Zipf frequency
    avg_word_lengths = []  # average word length (excluding punctuation marks)

    # readability metrics
    flesch_scores, flesch_ease = [], []
    flesch_kincaid_scores = []
    gunning_fog_scores = []
    coleman_liau_scores = []
    dale_chall_scores = []
    ari_scores = []
    linsear_write_scores = []
    spache_scores = []

    stimuli = pd.read_csv('data/stimuli.csv', sep='\t')

    for index, row in stimuli.iterrows():
        text = row['gen_seq_trunc']

        # text length
        text_length = len(text.split())
        text_lengths.append(text_length)

        # average word length in characters excluding punctuation marks
        char_ctr = 0
        for word in text.split():
           char_ctr += len(word.strip())
        avg_word_lenth = char_ctr / text_length
        avg_word_lengths.append(avg_word_lenth)

        # average word frequency, excluding punctuation marks
        word_freqs = [word_frequency(w.strip(), lang='en') for w in text.split()]
        avg_word_freqs.append(np.mean(np.array(word_freqs)))

        # average Zipf frequency, excluding punctuation marks
        zip_freqs = [zipf_frequency(w.strip(), lang='en') for w in text.split()]
        avg_zipf_freqs.append(np.mean(np.array(zip_freqs)))

        # readability metrics
        r = Readability(text, min_words=50)
        try:
            flesch_scores.append(r.flesch().score)
            flesch_ease.append(r.flesch().ease)
            flesch_kincaid_scores.append(r.flesch_kincaid().score)
            gunning_fog_scores.append(r.gunning_fog().score)
            coleman_liau_scores.append(r.coleman_liau().score)
            dale_chall_scores.append(r.dale_chall().score)
            ari_scores.append(r.ari().score)
            linsear_write_scores.append(r.linsear_write().score)
            spache_scores.append(r.spache().score)
        except:
            print('Cannot compute readability metric. At least 50 words required.')
            flesch_scores.append(np.nan)
            flesch_ease.append(np.nan)
            flesch_kincaid_scores.append(np.nan)
            gunning_fog_scores.append(np.nan)
            coleman_liau_scores.append(np.nan)
            dale_chall_scores.append(np.nan)
            ari_scores.append(np.nan)
            linsear_write_scores.append(np.nan)
            spache_scores.append(np.nan)

    # add text metrics to data frame
    stimuli['text_length'] = text_lengths
    stimuli['avg_word_freq'] = avg_word_freqs
    stimuli['avg_zip_freq'] = avg_zipf_freqs
    stimuli['avg_word_length'] = avg_word_lengths
    stimuli['flesch'] = flesch_scores
    stimuli['flesch_ease'] = flesch_ease
    stimuli['flesch_kincaid'] = flesch_kincaid_scores
    stimuli['gunning_fog'] = gunning_fog_scores
    stimuli['coleman_liau'] = coleman_liau_scores
    stimuli['dale_chall'] = dale_chall_scores
    stimuli['ari'] = ari_scores
    stimuli['linsear_write'] = linsear_write_scores
    stimuli['spache'] = spache_scores

    stimuli.to_csv('data/stimuli.csv', index=False, sep='\t')


if __name__ == '__main__':
    raise SystemExit(main())
