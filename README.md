# EMTeC: Eye movements on Machine-generated Texts Corpus

* link to OSF
* link to preprint

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


## Prompting of LLMs and text generation

In order to prompt Phi-2, Mistral and WizardLM, you can run the bash script

```bash
bash run_generation.sh 
```
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

