#!/usr/bin/env python3
"""

"""


import os
import os.path
from argparse import ArgumentParser

from typing import Dict, List, Any, Optional, Tuple

import torch
import numpy as np
import pandas as pd
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed, BitsAndBytesConfig

from generation.generation_utils import create_generation_config, generate_texts
from generation.generation_constants import tensor_path, cache_path


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--dec',
        type=str,
        help='The decoding strategy to be used.',
        choices=['greedy_search', 'beam_search', 'sampling', 'topk', 'topp'],
    )
    parser.add_argument(
        '--model',
        type=str,
        help='The model to prompt.',
        required=True,
        choices=['mistral', 'phi2', 'wizardlm'],
    )
    parser.add_argument(
        '--subset-idx',
        nargs=2,
        help='In case of CUDA OOM problems, run the model generation only on certain rows. Subset-idx indicates '
             'the rows of the prompts data frame to take into consideration. A Tuple of two integers, indicating '
             'a range, e.g., 0 36.',
        default=None,
        #default=[0, 66],
        type=int,
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=150,
    )
    parser.add_argument(
        '--k',
        type=int,
        default=50,
        help='The number of (most likely) words to retain for top-k sampling.'
    )
    parser.add_argument(
        '--seq-path',
        type=str,
        default='generation/output_generation/',
        help='The path to the dir where to save the output df.',
    )
    parser.add_argument(
        '--prompt-filename',
        type=str,
        default='prompts.csv',
        help='The name of the CSV file that contains the prompts.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=10,  # for re-generation: used seed 11
        help='The seed to use for reproducibility.',
    )
    parser.add_argument(
        '--items',
        nargs='+',
        type=str,
        default=None,
        help='The items to prompt for, e.g., item14 item15 item16.',
    )
    return parser


def main():
    os.environ['TRANSFORMERS_CACHE'] = cache_path
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    model = args.model
    decoding_strategy = args.dec
    model_name, template = '', ''
    eos_token, subword_delim, newline_token = '', '', ''

    ### load the appropriate model ###
    if model == 'mistral':
        model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
        template = '<s>[INST] {prompt} [/INST]'
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        eos_token = '</s>'
        subword_delim = '▁'
        newline_token = '<0x0A>'
    elif model == 'phi2':
        model_name = 'microsoft/phi-2'
        template = 'Instruct: {prompt}\nOutput:'
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype='auto',
        )
        newline_token = 'Ċ'
        subword_delim = 'Ġ'
        eos_token = '<|endoftext|>'
    elif model == 'wizardlm':
        model_name = 'WizardLM/WizardLM-13B-V1.2'
        template = 'A chat between a curious user and an artificial intelligence assistant. ' \
                   'The assistant gives helpful, detailed, and polite answers to the user\'s questions. ' \
                   'USER: {prompt} ASSISTANT:'
        if decoding_strategy == 'beam_search':
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map='auto',
                torch_dtype=torch.bfloat16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map='auto',
            )
        eos_token = '</s>'
        subword_delim = '▁'
        newline_token = '<0x0A>'
    else:
        raise NotImplementedError('This model is not intended for prompting.')

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # read in prompts
    prompts_df = pd.read_csv(os.path.join('generation', args.prompt_filename))

    if args.subset_idx:
        subset_idx1, subset_idx2 = args.subset_idx
        prompts_df = prompts_df.iloc[subset_idx1:subset_idx2]

    if args.items is not None:
        prompts_df = prompts_df[prompts_df['index'].isin(args.items)]

    # generation parameters that are identical across decoding strategies
    max_new_tokens = args.max_new_tokens
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.eos_token_id
    k = args.k
    return_dict_in_generate = True
    output_scores = True
    output_attentions = True
    output_hidden_states = True

    # generation config
    generation_config = create_generation_config(
        decoding_strategy=decoding_strategy,
        max_new_tokens=max_new_tokens,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        k=k,
        return_dict_in_generate=return_dict_in_generate,
        output_scores=output_scores,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )

    generate_texts(
        template=template,
        prompts_df=prompts_df,
        tokenizer=tokenizer,
        model=model,
        decoding_strategy=decoding_strategy,
        generation_config=generation_config,
        model_name=args.model,
        args=args,
        eos_token=eos_token,
        subword_delim=subword_delim,
        newline_token=newline_token,
        subset_idx=args.subset_idx,
        tensor_path=tensor_path,
        seq_path=args.seq_path,
    )



if __name__ == '__main__':
    raise SystemExit(main())
