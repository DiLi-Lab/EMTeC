# EMTeC: Eye movements on Machine-generated Texts Corpus

TODO update links
* The preprint **EMTeC: Eye movements on Machine-generated Texts Corpus** is available on TODO
* The eye-tracking data is available via [OSF](https://osf.io/ajqze/) or can be automatically downloaded using a python script (see below).
* The tensors are available on [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GCU0W8&version=DRAFT)
    * Attention: beware that the tensors require a lot of story to be downloaded. The following tensors are not provided via Dataverse because they exceed the maximum file size
        * Mistral: beam search item 34, item 43; greedy search item 43
        * WizardLM: beam search item 34, item 35, item 40, item 43; greedy search item 43; sampling item 43, top-k item 43, top-p item 43


This repository contains the code base used to create the experimental stimuli in EMTeC. It's for reproducibility. 
The entire dataset can be downloaded from OSF.

## Summary
TODO

## Setup

### Clone this repository
```bash
git clone git@github.com:dili-lab/emtec
```

### Install the requirements
The code is based on the PyTorch and huggingface modules.
```bash
pip install -r requirements.txt
```

## Download the data

### Eye-tracking data

**TODO make OSF repo public, adjust URLs**

The eye-tracking data is stored in an [OSF Repository](https://osf.io/ajqze/). To download and extract it directly. please run
```bash
python get_et_data.py --extract
```
which will automatically extract all zipped files. It will create a directory `EMTeC/data` that has the folder structure 
needed for reproducibility purposes.

### Tensors

**TODO make Dataset public, adjust URL in Readme and python script**
**TODO check if python script works once Dataset is public**

The transition scores, attention scores, hidden states, and beam indices are stored in a [Harvard Dataverse Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GCU0W8&version=DRAFT).
They can be downloaded from there or via calling
```bash
python get_tensors.py
```
***Attention:*** The tensors amount to about **340 GB** in size.



## Prompting of LLMs and text generation

In order to prompt Phi-2, Mistral and WizardLM, you can run the bash script


```bash
bash run_generation.sh 
```

**Note:** In order to prompt the models, you need GPUs set up with [CUDA](https://developer.nvidia.com/cuda-downloads). 

Beware that the GPUs are hard-coded in the bash script and depending on the kind of GPUs available, please 
adapt them accordingly. Moreover, running WizardLM with beam search as decoding strategy might result in CUDA OOM Errors 
for items that have a very long context. In such cases, you might subset the prompting process, as indicated within the 
bash script.

The text generation scripts not only prompt the models but they also post-process the output. Because the experimental 
stimuli presented in the eye-tracking experiment should follow certain criteria and formats, the model outputs are 
also truncated (in order to remove unfinished sentences and trailing whitespace characters) and cleaned.

During generation, the transition scores, beam indices, generated sequences, hidden states, and attention scores are 
saved to disk. Change the path name to the directory in which to save them accordingly in
`generation/generation_constants.py`.
Also adapt the path to the transformers cache in `generation/generation_constants.py`.

## Selecting and merging stimuli

Only a subset of the generated texts are presented in the eye-tracking experiment. In this step, the generated outputs 
are subset to the chosen items, which are found in `stimuli_selection/util_files/selected_items_0-105.npy`. Then the 
selected outputs of all models from all decoding strategies are merged. Each one is attributed to its list, the 
assignment of which can be found in `stimuli_selection/util_files/list_attribution.csv`, and the comprehension questions 
are added for each generated text, which can separately be found in `stimuli_selection/util_files/stimuli_and_questions_balanced.csv`.


```bash
bash run_stimuli_selection.sh
```


## Pre-Processing of Eye-Tracking Data

In order to reproduce the pre-processing of the eye-tracking data, place the subject directories into the `data` directory, 
the result of which should look like the following:
```
├── data
    ├── subject_level_data
    │   ├── ET_01
    │   │   ├── aoi
    │   │   ├── ET_01.edf
    │   │   ├── RESULTS_QUESTIONNAIRE.txt
    │   │   ├── RESULTS_QUESTIONS.txt
    │   ├── ET_02
    │   │   ├── aoi
    │   │   ├── ET_02.edf
    │   │   ├── RESULTS_QUESTIONNAIRE.txt
    │   │   ├── RESULTS_QUESTIONS.txt
    │   └── ...
    ├── stimuli_columns_descriptions.csv
    ├── stimuli.csv
```

### Conversion of `.edf` files to `.asc` files

The `.edf` files returned by the experimental software is not published. We nevertheless publish the code that implements 
this conversion. The result is an `.asc` file in each subject directory. Beware that the application that converts the `.edf` 
to `.asc` files only works in Linux- or Windows-based systems.


```bash
bash run_edf2asc.sh
```

```
├── data
    ├── subject_level_data
    │   ├── ET_01
    │   │   ├── aoi
    │   │   ├── ET_01.asc
    │   │   ├── ET_01.edf
    │   │   ├── RESULTS_QUESTIONNAIRE.txt
    │   │   ├── RESULTS_QUESTIONS.txt
    │   └── ...
    ├── stimuli_columns_descriptions.csv
    ├── stimuli.csv
```

### Conversion of `.asc` to `.csv` files

The `.asc` files are parsed and only the relevant information is extracted and written to csv files. While the `.edf` 
and the `.asc` files are not published, we nevertheless publish the code. To run the parsing of the `.asc` files, 
simply run
```bash
bash run_asc2csv.sh
```
The resulting data directory structure will then look like this:

```
├── data
    ├── subject_level_data
    │   ├── ET_01
    │   │   ├── aoi
    │   │   ├── ET_01.asc
    │   │   ├── ET_01.csv
    │   │   ├── ET_01.edf
    │   │   ├── RESULTS_QUESTIONNAIRE.txt
    │   │   ├── RESULTS_QUESTIONS.txt
    │   └── ...
    ├── stimuli_columns_descriptions.csv
    ├── stimuli.csv
```


### Fixation extraction

To group coordinate samples from the raw `.csv` files together into fixations and map them to the correct areas of 
interest (words), please run
```bash
bash run_csv2events.sh
```

The following arguments can be passed:
* `--disablle-parallel`: disable parallel processing
* `--plot-px-time`: if given, the raw x- and y-coordinates are plotted over time and the fixations extracted with the algorithm are marked.
* `--plot-ampl-vel`: if given, the peak saccade velocities are plotted over saccade amplitudes.
* `--threshold`: the threshold to use in the microsaccade detection algorithm. The default is `trial_based`, i.e. the threshold is estimated for each experimental stimulus individually.
* `--threshold-factor`: the factor with which the treshold is multiplied to obtain the radius
* `--threshold-method`: the method to compute the threshold
* `--min-fixation-duration-ms`: the minimum fixation duration in ms
* `--min-saccade-duration-ms`: the minimum saccade duration in ms
* `--max-saccade-velocity`: the maximum saccade velocity in deg/s
* `--theta`: the velocity threshold in deg/s

Since we publish the raw eye-tracking data, everything from this step onwards is reproducible.
For each subject, the resulting folder structure will then look like this (if both the coordinates over time as well as the 
saccade velocity over amplitude is plotted):


```
├── data
    ├── subject_level_data
    │   ├── ET_01
    │   │   ├── aoi
    │   │   ├── fixations
    │   │   │   ├── event_files
    │   │   │   │   ├── ET_01-item01-fixations.csv
    │   │   │   │   ├── ET_01-item02-fixations.csv
    │   │   │   │   └── ...
    │   │   │   ├── plots
    │   │   │   │   ├── ampl_vel
    │   │   │   │   │   ├── ET_01-item01-ampl_vel.png
    │   │   │   │   │   ├── ET_01-item01-ampl_vel_reg.png
    │   │   │   │   │   ├── ET_01-item02-ampl_vel.png
    │   │   │   │   │   ├── ET_01-item02-ampl_vel_reg.png
    │   │   │   │   │   └── ...
    │   │   │   │   ├── px_time
    │   │   │   │   │   ├── ET_01-item01-px_time.png
    │   │   │   │   │   ├── ET_01-item02-px_time.png
    │   │   │   │   │   └── ...
    │   │   │   ├── command_log.txt
    │   │   ├── ET_01.asc
    │   │   ├── ET_01.csv
    │   │   ├── ET_01.edf
    │   │   ├── RESULTS_QUESTIONNAIRE.txt
    │   │   ├── RESULTS_QUESTIONS.txt
    │   └── ...
    ├── stimuli_columns_descriptions.csv
    ├── stimuli.csv
```

The `event_files` directory contains the extracted fixations, one file per screen/experimental stimulus. The directory 
`ampl_vel` contains the plots of saccade amplitude over velocity, and `px_time` contains the coordinates over time plots. 
`command_log.txt` contains the programm call given in the bash script `run_csv2events.sh`.


### Fixation correction

The manual fixation correction, as opposed to all other preprocessing steps, requires different dependencies. 
In order to manually correct the extracted fixations, primarily fixing vertical drifts during eye-tracking, please first 
create a new virtual environment and install the necessary requirements:

```bash
pip install -r preprocessing/fixation_correction/fixcorr_requirements.txt
```
Then run the fixation correction script:
```bash
bash run_fixcorr.sh
```
The argument `--run-on-subj` indicates on which subject to run the fixation correction. If this argument is not given, the script 
will iterate through all files of all subjects. The resulting directory structure looks like this:

```
├── data
    ├── subject_level_data
    │   ├── ET_01
    │   │   ├── aoi
    │   │   ├── fixations
    │   │   │   ├── event_files
    │   │   │   │   ├── ET_01-item01-fixations.csv
    │   │   │   │   ├── ET_01-item02-fixations.csv
    │   │   │   │   └── ...
    │   │   │   ├── plots
    │   │   │   │   ├── ampl_vel
    │   │   │   │   │   ├── ET_01-item01-ampl_vel.png
    │   │   │   │   │   ├── ET_01-item01-ampl_vel_reg.png
    │   │   │   │   │   ├── ET_01-item02-ampl_vel.png
    │   │   │   │   │   ├── ET_01-item02-ampl_vel_reg.png
    │   │   │   │   │   └── ...
    │   │   │   │   ├── px_time
    │   │   │   │   │   ├── ET_01-item01-px_time.png
    │   │   │   │   │   ├── ET_01-item02-px_time.png
    │   │   │   │   │   └── ...
    │   │   │   ├── command_log.txt
    │   │   ├── fixations_corrected
    │   │   │   ├── event_files
    │   │   │   │   ├── ET_01-item01-fixations_corrected.csv
    │   │   │   │   ├── ET_01-item02-fixations_corrected.csv
    │   │   │   │   └── ...
    │   │   ├── ET_01.asc
    │   │   ├── ET_01.csv
    │   │   ├── ET_01.edf
    │   │   ├── RESULTS_QUESTIONNAIRE.txt
    │   │   ├── RESULTS_QUESTIONS.txt
    │   └── ...
    ├── stimuli_columns_descriptions.csv
    ├── stimuli.csv
```



### Computation of reading measures

To compute word-based reading measures from the fixation sequences, run

```bash
bash run_events2rms.sh
```
The reading measures are saved in one file per screen/experimental stimulus, and the resulting folder structure is 
indicated below. The keyword `--corrected` in the bash script indicates that computation of reading measures should only be 
done on corrected fixations. If it is omitted, reading measures are computed on the uncorrected fixations.
```
├── data
    ├── subject_level_data
    │   ├── ET_01
    │   │   ├── aoi
    │   │   ├── fixations
    │   │   │   ├── event_files
    │   │   │   │   ├── ET_01-item01-fixations.csv
    │   │   │   │   ├── ET_01-item02-fixations.csv
    │   │   │   │   └── ...
    │   │   │   ├── plots
    │   │   │   │   ├── ampl_vel
    │   │   │   │   │   ├── ET_01-item01-ampl_vel.png
    │   │   │   │   │   ├── ET_01-item01-ampl_vel_reg.png
    │   │   │   │   │   ├── ET_01-item02-ampl_vel.png
    │   │   │   │   │   ├── ET_01-item02-ampl_vel_reg.png
    │   │   │   │   │   └── ...
    │   │   │   │   ├── px_time
    │   │   │   │   │   ├── ET_01-item01-px_time.png
    │   │   │   │   │   ├── ET_01-item02-px_time.png
    │   │   │   │   │   └── ...
    │   │   │   ├── command_log.txt
    │   │   ├── fixations_corrected
    │   │   │   ├── event_files
    │   │   │   │   ├── ET_01-item01-fixations_corrected.csv
    │   │   │   │   ├── ET_01-item02-fixations_corrected.csv
    │   │   │   │   └── ...
    │   │   ├── reading_measures_corrected
    │   │   │   ├── ET_01-item01-reading_measures_corrected.csv
    │   │   │   ├── ET_01-item02-reading_measures_corrected.csv
    │   │   │   └── ...
    │   │   ├── ET_01.asc
    │   │   ├── ET_01.csv
    │   │   ├── ET_01.edf
    │   │   ├── RESULTS_QUESTIONNAIRE.txt
    │   │   ├── RESULTS_QUESTIONS.txt
    │   └── ...
    ├── stimuli_columns_descriptions.csv
    ├── stimuli.csv
```


## Lexical annotation

### Annotation on text-level

Please insert your huggingface access token in line 109 in `annotation/annotations.py`, then in order to annotate the stimuli texts with readability scores, please run
```bash
bash run_annotation_text.sh
```
The package `readability_local` is a local copy of `py-readability-metrics` (see their [Documentation](https://pypi.org/project/py-readability-metrics/) and their [GitHub](https://github.com/cdimascio/py-readability-metrics/tree/master)),
as I had to adjust the minimum number of words. This annotation will directly add the readability metrics to the `stimuli.csv` file.

### Annotation on word-level

To annotate the stimuli texts on word-level with frequency scores, PoS tags, dependency tags, surprisal values, etc., please run the bash script

```bash
bash run_annotation_word.sh gpt2 gpt2-large opt-350m opt-1.3b mistral-base mistral-instruct phi2 llama2-7b llama2-13b pythia-6.9b pythia-12b
```
This will first create a folder `unique_aois` in the `data` directory, which contains the areas of interest on word-level 
for each stimulus text (i.e., each condition; item id, model, decoding strategy). First the PoS tags, dependency tags, 
word length information, and frequency values are added to a temporary file `annotation/temp_annotated_data.csv`. Then 
the surprisal values are extracted from the language models specified in the bash call and the final annotations are 
saved in `annotations/word_level_annotations.csv`.

**Note:** In order to prompt the models, you need GPUs set up with [CUDA](https://developer.nvidia.com/cuda-downloads).




### Merging files

In order to merge 
* the fixation sequences
* the corrected fixation sequences
* the reading measures
* the corrected reading measures
* the participant information (questionnaire and comprehension question information), including extracting the information on which eye was tracked for each participant
* the word-level lexical annotation with the reading measures
* the word-level lexical annotation with the corrected reading measures

please run 
```bash
bash run_merge.sh
```
The word-level annotations are merged directly into the files `data/reading_measures.csv` and `data/reading_measures_corrected.csv`.
The resulting folder structure will then look like this:

```
├── data
    ├── participant_info
    │   ├── participant_info.csv
    │   ├── participant_results.csv
    ├── subject_level_data
    │   ├── ET_01
    │   │   ├── aoi
    │   │   ├── fixations
    │   │   │   ├── event_files
    │   │   │   │   ├── ET_01-item01-fixations.csv
    │   │   │   │   ├── ET_01-item02-fixations.csv
    │   │   │   │   └── ...
    │   │   │   ├── plots
    │   │   │   │   ├── ampl_vel
    │   │   │   │   │   ├── ET_01-item01-ampl_vel.png
    │   │   │   │   │   ├── ET_01-item01-ampl_vel_reg.png
    │   │   │   │   │   ├── ET_01-item02-ampl_vel.png
    │   │   │   │   │   ├── ET_01-item02-ampl_vel_reg.png
    │   │   │   │   │   └── ...
    │   │   │   │   ├── px_time
    │   │   │   │   │   ├── ET_01-item01-px_time.png
    │   │   │   │   │   ├── ET_01-item02-px_time.png
    │   │   │   │   │   └── ...
    │   │   │   ├── command_log.txt
    │   │   ├── fixations_corrected
    │   │   │   ├── event_files
    │   │   │   │   ├── ET_01-item01-fixations_corrected.csv
    │   │   │   │   ├── ET_01-item02-fixations_corrected.csv
    │   │   │   │   └── ...
    │   │   ├── reading_measures_corrected
    │   │   │   ├── ET_01-item01-reading_measures_corrected.csv
    │   │   │   ├── ET_01-item02-reading_measures_corrected.csv
    │   │   │   └── ...
    │   │   ├── ET_01.asc
    │   │   ├── ET_01.csv
    │   │   ├── ET_01.edf
    │   │   ├── RESULTS_QUESTIONNAIRE.txt
    │   │   ├── RESULTS_QUESTIONS.txt
    │   └── ...
    ├── unique_aois
    ├── fixations.csv
    ├── fixations_corrected.csv
    ├── reading_measures.csv
    ├── reading_measures_corrected.csv
    ├── stimuli_columns_descriptions.csv
    ├── stimuli.csv
```


### Psycholinguistic analyses

To run the psycholinguistic analyses please run the regression models for the different response variables (first-pass reading time, first-pass regression etc.) using the following command.

```bash
mkdir model_fits, logs
bash run_regression_models.sh
```

This will save the brms-fits in the folder `model_fits`.
To extract and plot the posterior distributions of the parameters of interest (word length, surprisal, etc.), please run 
```bash
bash Rscript --vanilla analyses/extract_and_plot.R
```