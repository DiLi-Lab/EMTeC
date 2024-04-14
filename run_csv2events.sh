#!/bin/sh

python3 -m preprocessing.utils.create_aoi_csv

python3 -m preprocessing.parsing.extract_events \
--threshold-factor 2.5 \
--threshold-method engbert2015 \
--plot-px-time   \
--plot-ampl-vel  \
#--disable-parallel \
#--threshold trial_based #> events_error.log 2>&1
