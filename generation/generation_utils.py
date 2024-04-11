"""

"""

import os
import os.path
import json
import string
import datetime

import argparse
from argparse import ArgumentParser

from typing import Dict, List, Any, Optional, Tuple, Union

import torch
import numpy as np
import pandas as pd
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed, BitsAndBytesConfig



def get_cut_idx(
        scores: Tuple[torch.FloatTensor],
        sequences: torch.LongTensor,
        beam_indices: Optional[torch.Tensor] = None,
) -> int:
    if beam_indices is None:
        beam_indices = torch.arange(scores[0].shape[0]).view(-1, 1)
        beam_indices = beam_indices.expand(-1, len(scores))
    beam_indices_mask = beam_indices < 0
    max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
    cut_idx = sequences.shape[-1] - max_beam_length
    return cut_idx


def create_generation_config(
        decoding_strategy: str,
        max_new_tokens: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        k: int,
        return_dict_in_generate: bool = True,
        output_scores: bool = True,
        output_attentions: bool = True,
        output_hidden_states: bool = True,
) -> GenerationConfig:
    generation_configs = {
        'greedy_search': GenerationConfig(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            max_new_tokens=max_new_tokens,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            do_sample=False,
            early_stopping=False,
            num_beams=1,
        ),
        'sampling': GenerationConfig(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            max_new_tokens=max_new_tokens,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            do_sample=True,
            top_k=0,
            temperature=0.8,
        ),
        'topk': GenerationConfig(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            max_new_tokens=max_new_tokens,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            do_sample=True,
            top_k=k,
            temperature=0.8,
        ),
        'topp': GenerationConfig(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            max_new_tokens=max_new_tokens,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
        ),
        'beam_search': GenerationConfig(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            max_new_tokens=max_new_tokens,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            do_sample=False,
            num_beams=4,
            early_stopping=False,
        ),
    }
    return generation_configs[decoding_strategy]


def generate_texts(
        template: str,
        prompts_df: pd.DataFrame,
        tokenizer: transformers.AutoTokenizer,
        model: transformers.AutoModelForCausalLM,
        decoding_strategy: str,
        generation_config: transformers.GenerationConfig,
        model_name: str,
        args: argparse.Namespace,
        eos_token: str,
        subword_delim: str,
        newline_token: str,
        subset_idx: Union[List[int], None],
        tensor_path: str,
        seq_path: str,
):
    today = datetime.date.today().strftime('%Y-%m-%d')
    output = {
        'item_id': list(),                          # the unique id of the item/prompt
        'task': list(),                             # story generation, synopsis, article, dialogue, ...
        'type': list(),                             # creative writing vs. constrained
        'subcategory': list(),
        'decoding_strategy': list(),                # sampling, topk, topp, greedy_search, beam_search
        'generation_config': list(),
        'prompt': list(),                           # the prompt
        'prompt_toks': list(),                      # the tokenized prompt as input to the model
        'prompt_input_ids': list(),                 # the input ids of the prompt as input to the model
        'gen_seq': list(),                          # the generated sequence, excluding the prompt
        'gen_seq_trunc': list(),                    # the truncated generated sequence (i.e., without incomplete sentences and trailing newlines)
        'gen_toks': list(),                         # the generated subword tokens
        'gen_toks_trunc': list(),                   # the generated subword tokens excluding unfinished sents and trailing newlines
        'gen_toks_trunc_wo_nl': list(),             # the generated subword tokens excl. unfinished sents, trailing newlines and in-text newlines
        'gen_toks_trunc_wo_nl_wo_punct': list(),    # the generated subword tokens excl. unfinished sents, trailing \n, in-text \n and punctuation marks
        'gen_ids': list(),                          # the generated ids (input ids) of the model
        'gen_ids_trunc': list(),                    # the generated input ids without trailing newlines and incomplete sentences
        'gen_ids_trunc_wo_nl': list(),              # the generated input ids without trailing newlines, incomplete sents, and in-text newlines
        'gen_ids_trunc_wo_nl_wo_punct': list(),     # the generated input ids without trailing newlines, incomplete sents, in-text newlines and punctuation marks
        'tok_idx': list(),                          # unique id for every generated token, before truncation
        'tok_idx_trunc': list(),                    # the unique ids for the truncated text (without unfinished sentences and trailing newlines)
        'tok_idx_trunc_wo_nl': list(),              # the unique ids subset to the tokens excluding newlines within the text (used when subsetting the original scores, entropies etc. to the truncated ones)
        'tok_idx_trunc_wo_nl_wo_punct': list(),     # the unique ids subset to the tokens excluding newlines within the text and punctuation marks (used when subsetting the original scores, entropies etc. to the truncated ones)
        'truncated_original': list(),               # whether the original text was truncated (i.e., was there an unfinished sentence or trailing nl?)
        'removed_newlines': list(),                 # whether there were in-text newlines (that were removed)
        'word_ids_list_wo_nl': list(),              # list of word ids that map subword tokens together. excluding in-text newlines (because we won't have eye-tracking measures for newlines). aligns when splitting words on whitespace (as will be during ET)
        'word_ids_list_wo_nl_wo_punct': list(),     # word ids that map subword tokens together, excluding in-text newlines and punctuation marks. aligns when splitting words on whitespace because punct always attached to word (as will be during ET)
        'gen_seq_trunc_split_wl': list(),           # the truncated sequence (w/o trailing nl and unfinished sents) split on whitespace; what will be the interest areas in the ET
                                                    #    newline tokens are kind of in there, otherwise the splitting will not work. alignment works with the word ids.
        'alignment_mismatch': list(),               # whether there is an error when comparing the length of the split sentence (word level) with the number of word ids
        'remove_ctr': list(),                       # the index for truncating generated sequences (unfinished sentences, eos token, trailing newlines)
        'cut_nl_idx': list(),                       # only for poems; cut them after certain number of newlines
    }

    item_ids = list()

    for row_index, row in prompts_df.iterrows():

        # for the 30b+ models, item 44 will result in a cuda out of memory error, even when distributing across all 8
        # GPUs (somehow beam search still loads most onto the first GPU)
        if model_name in ['mpt', 'llama30', 'llama70'] and decoding_strategy == 'beam_search' and item_id in ['item43', 'item44']:
            continue



        instance = row['prompt']
        item_id = row['index']
        item_ids.append(item_id)
        task = row['task']
        type = row['type']
        subcategory = row['subcategory']

        # if subcagetory is NaN, convert to string so type matches with other entries
        if not isinstance(subcategory, str):
            subcategory = str(subcategory)

        # tokenize and generate
        print(f'--- decoding {decoding_strategy}\titem {item_id}')
        prompt = template.format(prompt=instance)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        out = model.generate(**inputs, generation_config=generation_config, pad_token_id=tokenizer.eos_token_id)

        # cut off the prompt from the generated sequence
        if decoding_strategy != 'beam_search':
            beam_indices = None
        else:
            beam_indices = out.beam_indices
        cut_idx = get_cut_idx(scores=out.scores, sequences=out.sequences, beam_indices=beam_indices)
        gen_seq = tokenizer.decode(out.sequences[0][cut_idx:].cpu())
        gen_toks = tokenizer.convert_ids_to_tokens(out.sequences[0][cut_idx:].cpu())
        gen_ids = out.sequences[:, cut_idx:].cpu().tolist()
        tok_idx = list(range(len(gen_toks)))
        assert len(tok_idx) == len(gen_ids[0]) == len(gen_toks)

        ####### POSTPROCESSING #######

        ##### Truncation #####
        ### remove EOS tokens, trailing newlines, truncate sequences that end mid-sentence ###
        remove_ctr = 0
        truncated_original = 'yes'
        if not any(tok in '".!?' for tok in gen_toks):
            # account for edge case where there is not a single end-of-sentence punctuation mark predicted (poem)
            # remove EOS token and trailing newline
            if gen_toks[-1] == eos_token:
                remove_ctr += 1
                if gen_toks[-2] == newline_token:
                    remove_ctr += 1
            else:
                truncated_original = 'no'
        else:
            # cut away the part following the last end-of-sentence punctuation mark
            for idx, tok in enumerate(reversed(gen_toks)):
                if tok in '".!?':
                    if idx == 0:
                        truncated_original = 'no'
                    # account for cases where there is an enumeration (would stop after '2.')
                    if gen_toks[-(idx+2)] in '123456789':
                        remove_ctr += 1
                        continue
                    # account for cases where it would end on a quotation mark indicating the beginning of a turn'
                    if tok == '"' and gen_toks[-(idx+2)] == newline_token:
                        remove_ctr += 1
                        continue
                    if tok == '"' and gen_toks[-(idx+2)] not in '.!?':
                        remove_ctr += 1
                        continue
                    break
                else:
                    remove_ctr += 1
        # if the very last token is an end of sentence punctuation mark, the remove ctr is 0 --> retain everything
        if remove_ctr == 0:
            gen_toks_trunc = gen_toks[:]
            gen_ids_trunc = [gen_ids[0][:]]
            gen_seq_trunc = tokenizer.decode(gen_ids_trunc[0])
            tok_idx_trunc = tok_idx[:]
        # else cut off unfinished part
        else:
            gen_toks_trunc = gen_toks[:-remove_ctr]
            gen_ids_trunc = [gen_ids[0][:-remove_ctr]]
            gen_seq_trunc = tokenizer.decode(gen_ids_trunc[0])
            tok_idx_trunc = tok_idx[:-remove_ctr]


        ### poems ###
        # for poems, restrict the maximum number of newlines, otherwise it will not fit onto the ET screen
        cut_nl_idx = 0
        if task == 'poetry':
            # cut off after 9 newlines, if it's the end of a verse
            nl_ctr = 0
            # account for edge case when there is not a single end of sentence punctuation mark predicted
            # cut off after maximal 9 newlines
            if not any(tok in '".!?' for tok in gen_toks):
                for idx, tok in enumerate(gen_toks_trunc):
                    if tok == newline_token:
                        nl_ctr += 1
                        cut_nl_idx = idx
                    if nl_ctr >= 9:
                        break
                    if idx == len(gen_toks_trunc) - 1:
                        cut_nl_idx = len(gen_toks_trunc)
            # else cut of after maximal 9 newlines, if the previous line ends with an end-of-sentence punctuation mark
            else:
                for idx, tok in enumerate(gen_toks_trunc):
                    if tok == newline_token:
                        nl_ctr += 1
                        cut_nl_idx = idx

                    if model_name == 'phi2':
                        if nl_ctr >= 9:
                            if gen_toks_trunc[idx - 1] == subword_delim and gen_toks_trunc[idx - 2] in '.!?':

                                break
                            if gen_toks_trunc[idx - 1] in '.!?':
                                break

                    else:
                        if nl_ctr >= 9 and gen_toks_trunc[idx - 1] in '.!?':
                            break
                    if idx == len(gen_toks_trunc) - 1 and gen_toks_trunc[idx] in '.!?':
                        cut_nl_idx = len(gen_toks_trunc)
            gen_toks_trunc = gen_toks_trunc[:cut_nl_idx]
            gen_ids_trunc = [gen_ids_trunc[0][:cut_nl_idx]]
            gen_seq_trunc = tokenizer.decode(gen_ids_trunc[0])
            tok_idx_trunc = tok_idx_trunc[:cut_nl_idx]

        ##### Clean data #####

        # clean the data so that it can be used for an eye-tracking experiment
        # remove newlines and do the mapping to word ids
        # 1) map subword tokens to word ids so that they can be mapped to the correct words as well as scores of two
        #   or more subwords; that way, scores can also be correctly combined in joint probability, joint entropy, ...
        # 2) remove newlines from the subword tokens as well as from the scores
        #   the predicted newlines will still be presented in the ET experiment (i.e., we will adhere to the model's
        #   predicted layout/line breaks) but there will be no corresponding ET measure so we will not take newline
        #   scores into account. The same goes for predicted whitespaces, such as '\r' and '▁▁' and '   '
        # 3) remove punctuation marks: for the possibility of comparing the probabilities of words with the obtained
        #   eye-tracking measures but without taking into account punctuation probabilities
        # 4) account for other problems with generated sequences

        # 1) word ids
        # theoretically, all subwords that constitute the first subword of a new word should be preceded by a subword
        # delimiter, but this is not always the case, so if I want to group subwords together and attribute them the
        # same word ids, I need to account for edge cases and wrong predictions
        # example (from mistral): '▁photographs', '.', '<0x0A>', '<0x0A>', 'Em', 'ma', '▁was'
        #   Emma will be given the same id as photographs because there is no subword delimiter
        gen_toks_trunc_underscore = gen_toks_trunc.copy()
        for i in range(len(gen_toks_trunc_underscore)):
            if i > 0:
                # if model_name == 'mpt' and gen_toks_trunc_underscore[i] not in [newline_token, '\r'] and \
                #         gen_toks_trunc_underscore[i - 1] == '  ':
                if model_name == 'mpt' and gen_toks_trunc_underscore[i] == '  ':
                    gen_toks_trunc_underscore[i] = subword_delim + gen_toks_trunc_underscore[i]
                if i == 1:
                    if gen_toks_trunc_underscore[i] not in [newline_token, '\r'] and gen_toks_trunc_underscore[
                        i - 1] == newline_token:
                        gen_toks_trunc_underscore[i] = subword_delim + gen_toks_trunc_underscore[i]
                if i == 2:
                    if gen_toks_trunc_underscore[i] not in [newline_token, '\r'] and gen_toks_trunc_underscore[
                        i - 1] == newline_token and gen_toks_trunc_underscore[i - 2] == newline_token:
                        gen_toks_trunc_underscore[i] = subword_delim + gen_toks_trunc_underscore[i]
                    if gen_toks_trunc_underscore[i] not in [newline_token, '\r'] and gen_toks_trunc_underscore[
                        i - 1] == newline_token and gen_toks_trunc_underscore[i - 2] != newline_token and \
                            gen_toks_trunc_underscore[i - 2] != subword_delim:
                        gen_toks_trunc_underscore[i] = subword_delim + gen_toks_trunc_underscore[i]
                if i > 2:
                    if gen_toks_trunc_underscore[i] not in [newline_token, '\r'] and gen_toks_trunc_underscore[
                        i - 1] == newline_token and gen_toks_trunc_underscore[i - 2] == newline_token and \
                            gen_toks_trunc_underscore[i - 3] != subword_delim:
                        gen_toks_trunc_underscore[i] = subword_delim + gen_toks_trunc_underscore[i]
                    if gen_toks_trunc_underscore[i] not in [newline_token, '\r'] and gen_toks_trunc_underscore[
                        i - 1] == newline_token and gen_toks_trunc_underscore[i - 2] != newline_token and \
                            gen_toks_trunc_underscore[i - 2] != subword_delim:
                        gen_toks_trunc_underscore[i] = subword_delim + gen_toks_trunc_underscore[i]

        # create list of word IDs (that map subwords together) based on tokens starting with underscore
        # problem: sometimes, especially when indicating the beginning of numbers, the subword delimiter is a
        # separate token. example: '▁sp', 'anned', '▁from', '▁', '1', '7', '5', '0', '▁to', '▁', '1', '8', '2', '0',
        word_ids_list = list()
        current_id = 0
        for i, tok in zip(tok_idx_trunc, gen_toks_trunc_underscore):
            # account for several subword delimiters meaning whitespace
            if tok.startswith(subword_delim) and not tok in ['▁▁', '▁▁▁']:
                if i == 0:
                    word_ids_list.append(current_id)
                else:
                    # account for edge case when model (falcon) predicts first a newline and
                    # gen toks trunc underscore looks like this: ['Ċ', 'ĠAs', 'Ġthe', 'Ġperson', 'Ġwalked' ...
                    if model_name == 'falcon' and i == 1 and \
                            gen_toks_trunc_underscore[0] == newline_token and tok.startswith(subword_delim):
                        word_ids_list.append(current_id)
                    else:
                        current_id += 1
                        word_ids_list.append(current_id)
            else:
                word_ids_list.append(current_id)

        # 2) subset the word ids and the token indices to those without newlines and whitespace characters
        # also remove stand-alone subword delimiters
        # tok_idx_trunc_wo_nl is the indices in the original gen_toks, not the indices of the toks in gen_toks_nl
        tok_idx_trunc_wo_nl = [tok_idx for tok_idx, tok in zip(tok_idx_trunc, gen_toks_trunc) if
                               tok not in [newline_token, subword_delim, '▁▁', '▁▁▁']]
        word_ids_list_wo_nl = [word_id for word_id, tok in zip(word_ids_list, gen_toks_trunc) if
                               tok not in [newline_token, subword_delim, '▁▁', '▁▁▁']]
        gen_toks_trunc_wo_nl = [tok for tok in gen_toks_trunc if tok not in [newline_token, subword_delim, '▁▁', '▁▁▁']]
        gen_ids_trunc_wo_nl = [gid for gid, tok in zip(gen_ids_trunc[0], gen_toks_trunc) if
                               tok not in [newline_token, subword_delim, '▁▁', '▁▁▁']]
        assert (len(tok_idx_trunc_wo_nl) == len(word_ids_list_wo_nl) == len(gen_toks_trunc_wo_nl) == len(
            gen_ids_trunc_wo_nl))
        removed_newlines = 'no'
        if len(gen_toks_trunc_wo_nl) < len(gen_toks_trunc):
            removed_newlines = 'yes'
        # the unique list of word IDs (set) will match when splitting gen_toks_trunc.split(), as it will split away
        # the newlines and whitespace characters

        # 3) remove punctuation
        # we only remove punctuation marks that are attached to words; 'free-standing' punctuation marks are kept
        # e.g., in dialogue, the model might produce newlines then: '- Alice: text...', '- Bob: text...'
        punct = string.punctuation
        tok_idx_trunc_wo_nl_wo_punct = []
        word_ids_list_wo_nl_wo_punct = []
        gen_toks_trunc_wo_nl_wo_punct = []
        gen_ids_trunc_wo_nl_wo_punct = []
        for idx in range(len(gen_toks_trunc_wo_nl)):
            # if token is not a punctuation mark: keep
            if gen_toks_trunc_wo_nl[idx] not in punct:
                tok_idx_trunc_wo_nl_wo_punct.append(tok_idx_trunc_wo_nl[idx])
                word_ids_list_wo_nl_wo_punct.append(word_ids_list_wo_nl[idx])
                gen_toks_trunc_wo_nl_wo_punct.append(gen_toks_trunc_wo_nl[idx])
                gen_ids_trunc_wo_nl_wo_punct.append(gen_ids_trunc_wo_nl[idx])
            # else the token is a punctuation mark
            # keep the punctuation mark if it stands alone, i.e., if it has its own word id
            else:
                if idx == 0:
                    if word_ids_list_wo_nl[idx] != word_ids_list_wo_nl[idx + 1]:
                        tok_idx_trunc_wo_nl_wo_punct.append(tok_idx_trunc_wo_nl[idx])
                        word_ids_list_wo_nl_wo_punct.append(word_ids_list_wo_nl[idx])
                        gen_toks_trunc_wo_nl_wo_punct.append(gen_toks_trunc_wo_nl[idx])
                        gen_ids_trunc_wo_nl_wo_punct.append(gen_ids_trunc_wo_nl[idx])
                    else:
                        continue
                elif idx == len(gen_toks_trunc_wo_nl)-1:
                    if word_ids_list_wo_nl[idx] != word_ids_list_wo_nl[idx - 1]:
                        tok_idx_trunc_wo_nl_wo_punct.append(tok_idx_trunc_wo_nl[idx])
                        word_ids_list_wo_nl_wo_punct.append(word_ids_list_wo_nl[idx])
                        gen_toks_trunc_wo_nl_wo_punct.append(gen_toks_trunc_wo_nl[idx])
                        gen_ids_trunc_wo_nl_wo_punct.append(gen_ids_trunc_wo_nl[idx])
                    else:
                        continue
                else:
                    if (word_ids_list_wo_nl[idx] != word_ids_list_wo_nl[idx - 1]
                            and word_ids_list_wo_nl[idx] != word_ids_list_wo_nl[idx + 1]):
                        tok_idx_trunc_wo_nl_wo_punct.append(tok_idx_trunc_wo_nl[idx])
                        word_ids_list_wo_nl_wo_punct.append(word_ids_list_wo_nl[idx])
                        gen_toks_trunc_wo_nl_wo_punct.append(gen_toks_trunc_wo_nl[idx])
                        gen_ids_trunc_wo_nl_wo_punct.append(gen_ids_trunc_wo_nl[idx])
                    else:
                        continue
        assert len(tok_idx_trunc_wo_nl_wo_punct) == len(word_ids_list_wo_nl_wo_punct) == len(
            gen_toks_trunc_wo_nl_wo_punct) == len(gen_ids_trunc_wo_nl_wo_punct)

        # 4) other problems
        # if "Ġ'" is in the generated tokens, i.e. id 705, the decode method does not properly decode it
        # 'Ġco', 'ining', 'Ġthe', 'Ġterm', "Ġ'", 's', 'oci', 'ology', "'", 'Ġin', 'Ġthe is decoded into
        # 'coining the term'sociology' in the', which results in word id alignment mismatch
        # in the case of Ġ' present, add another empty space to the generated ids, but only use these ids for decoding
        if subword_delim == 'Ġ':
            if 705 in gen_ids[0]:
                index = gen_ids[0].index(705)
                # 220 is the input id of Ġ
                gen_ids_whitespace = gen_ids.copy()
                gen_ids_whitespace[0].insert(index, 220)
                gen_seq = tokenizer.decode(gen_ids_whitespace[0])
            if 705 in gen_ids_trunc[0]:
                index = gen_ids_trunc[0].index(705)
                # 220 is the input id of Ġ
                gen_ids_trunc_whitespace = gen_ids_trunc.copy()
                gen_ids_trunc_whitespace[0].insert(index, 220)
                gen_seq_trunc = tokenizer.decode(gen_ids_trunc_whitespace[0])


        ##### Pooling #####
        # pooling to word-level (for eye-tracking experiment); wl = word-level
        # split the generated truncated sequence (will get rid of newliens and will match with the generated
        # tokens and scores where we cut off newlines and whitespace characters)
        gen_seq_trunc_split_wl = gen_seq_trunc.split()
        # make sure there are is no bug (i.e., there are as many words as there are unique word IDs)
        alignment_mismatch = 'no'
        try:
            assert len(set(word_ids_list_wo_nl)) == len(gen_seq_trunc_split_wl)
        except AssertionError:
            print(
                f'#### ERROR {model_name} {decoding_strategy} {item_id} '
                f'assert len(set(word_ids_list_wo_nl)) == len(gen_seq_trunc_split_wl)'
            )
            alignment_mismatch = 'yes'
            breakpoint()
        # same for the ids where punctuation marks are removed (because they all belong to some words)
        try:
            assert len(set(word_ids_list_wo_nl_wo_punct)) == len(gen_seq_trunc_split_wl)
        except AssertionError:
            print(
                f'#### ERROR {model_name} {decoding_strategy} {item_id} '
                f'assert len(set(word_ids_list_wo_nl_wo_punct)) == len(gen_seq_trunc_split_wl)'
            )
            alignment_mismatch = 'yes'


        ####### SAVE MODEL OUTPUTS #######
        # phi-2 does not return attention scores and hidden states
        path_to_save = os.path.join(tensor_path, model_name)
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        base_filename = f'{model_name}_{decoding_strategy}_{item_id}'

        ### save the original scores, sequences, etc. ###
        # sequences: account for the prompt being also in the output
        torch.save(out.sequences.cpu(), os.path.join(path_to_save, base_filename + '_sequences.pt'))
        # transition scores
        torch.save(tuple(tensor.cpu() for tensor in out.scores),
                   os.path.join(path_to_save, base_filename + '_scores.pt'))
        if not model_name == 'phi2':
            # attention scores
            torch.save(tuple(tuple(tensor.to('cpu') for tensor in inner_tuple) for inner_tuple in out.attentions),
                       os.path.join(path_to_save, base_filename + '_attentions.pt'))
            # hidden states
            torch.save(tuple(tuple(tensor.to('cpu') for tensor in inner_tuple) for inner_tuple in out.hidden_states),
                       os.path.join(path_to_save, base_filename + '_hidden_states.pt'))
        if decoding_strategy == 'beam_search':
            # save beam
            torch.save(beam_indices.cpu(), os.path.join(path_to_save, base_filename + '_beam_indices.pt'))

        ### save the truncated tensors (without unfinished sentences, trailing newlines, ..) ##
        # if generated sequence was not truncated, the original tensor is saved
        # beam indices
        if decoding_strategy == 'beam_search':
            # the end of the beam indices is given by the prompt; so we want to keep the generated part up until we
            # have truncated (unfinished sentences + trailing eos + nl) plus the part referring to the prompt at the end
            beam_indices_trunc = torch.cat(
                [beam_indices[0][tok_idx_trunc], beam_indices[0][-len(inputs['input_ids'][0]):]]
            )
            torch.save(beam_indices_trunc.cpu(), os.path.join(path_to_save, base_filename + '_beam_indices_trunc.pt'))
        # transition scores
        torch.save(tuple(tensor.cpu() for tensor in out.scores[:len(tok_idx_trunc)]),
                   os.path.join(path_to_save, base_filename + '_scores_trunc.pt'))
        # sequences: account for the prompt being also in the output
        torch.save(out.sequences[:, :len(inputs['input_ids'][0]) + len(tok_idx_trunc)].cpu(),
                   os.path.join(path_to_save, base_filename + '_sequences_trunc.pt'))
        if not model_name == 'phi2':
            # attention scores
            torch.save(tuple(tuple(tensor.to('cpu') for tensor in inner_tuple) for inner_tuple in
                             out.attentions[:len(tok_idx_trunc)]),
                       os.path.join(path_to_save, base_filename + '_attentions_trunc.pt'))
            # hidden states
            torch.save(tuple(tuple(tensor.to('cpu') for tensor in inner_tuple) for inner_tuple in
                             out.hidden_states[:len(tok_idx_trunc)]),
                       os.path.join(path_to_save, base_filename + '_hidden_states_trunc.pt'))

        ### save the tensors where newlines, whitespaces and weird predicted underscores are removed ###
        # transition scores
        torch.save(tuple(out.scores[i].to('cpu') for i in tok_idx_trunc_wo_nl),
                   os.path.join(path_to_save, base_filename + '_scores_trunc_wo_nl.pt'))
        # sequences: account for the prompt being also in the output
        shifted_indices = list(map(lambda x: x + len(inputs['input_ids'][0]), tok_idx_trunc_wo_nl))
        torch.save(out.sequences[:, shifted_indices].cpu(),
                   os.path.join(path_to_save, base_filename + '_sequences_trunc_wo_nl.pt'))
        # beam indices
        if decoding_strategy == 'beam_search':
            # end of beam indices refers to prompt; keep prompt and truncated beam indices minus those referring to \n
            beam_indices_trunc_wo_nl = torch.cat([beam_indices[0][tok_idx_trunc_wo_nl],
                                                  beam_indices[0][-len(inputs['input_ids'][0]):]])
            torch.save(beam_indices_trunc_wo_nl.cpu(),
                       os.path.join(path_to_save, base_filename + '_beam_indices_trunc_wo_nl.pt'))
        if not model_name == 'phi2':
            # attention scores
            torch.save(tuple(tuple(tensor.to('cpu') for tensor in out.attentions[i]) for i in tok_idx_trunc_wo_nl),
                       os.path.join(path_to_save, base_filename + '_attentions_trunc_wo_nl.pt'))
            # hidden states
            torch.save(tuple(tuple(tensor.to('cpu') for tensor in out.hidden_states[i]) for i in tok_idx_trunc_wo_nl),
                       os.path.join(path_to_save, base_filename + '_hidden_states_trunc_wo_nl.pt'))

        ### save the tensors where the above-mentioned things plus punctuation marks are removed
        # transition scores
        torch.save(tuple(out.scores[i].to('cpu') for i in tok_idx_trunc_wo_nl_wo_punct),
                   os.path.join(path_to_save, base_filename + '_scores_trunc_wo_nl_wo_punct.pt'))
        if not model_name == 'phi2':
            # attention scores
            torch.save(
                tuple(tuple(tensor.to('cpu') for tensor in out.attentions[i]) for i in tok_idx_trunc_wo_nl_wo_punct),
                os.path.join(path_to_save, base_filename + '_attentions_trunc_wo_nl_wo_punct.pt'))
            # hidden states
            torch.save(
                tuple(tuple(tensor.to('cpu') for tensor in out.hidden_states[i]) for i in tok_idx_trunc_wo_nl_wo_punct),
                os.path.join(path_to_save, base_filename + '_hidden_states_trunc_wo_nl_wo_punct.pt'))
        # sequences: account for the prompt being also in the output
        shifted_indices_wo_punct = list(map(lambda x: x + len(inputs['input_ids'][0]), tok_idx_trunc_wo_nl_wo_punct))
        torch.save(out.sequences[:, shifted_indices_wo_punct].cpu(),
                   os.path.join(path_to_save, base_filename + '_sequences_trunc_wo_nl_wo_punct.pt'))
        # beam indices
        if decoding_strategy == 'beam_search':
            # end of beam indices refers to prompt; keep prompt and truncated beam indices minus those referring to \n
            beam_indices_trunc_wo_nl_wo_punct = torch.cat([beam_indices[0][tok_idx_trunc_wo_nl_wo_punct],
                                                  beam_indices[0][-len(inputs['input_ids'][0]):]])
            torch.save(beam_indices_trunc_wo_nl_wo_punct.cpu(),
                       os.path.join(path_to_save, base_filename + '_beam_indices_trunc_wo_nl_wo_punct.pt'))


        ####### write generated to dict and file #######

        del out
        output['item_id'].append(item_id)
        output['task'].append(task)
        output['type'].append(type)
        output['subcategory'].append(subcategory)
        output['decoding_strategy'].append(decoding_strategy)
        output['generation_config'].append(generation_config.to_dict())
        output['prompt'].append(prompt)
        output['prompt_toks'].append(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
        output['prompt_input_ids'].append(inputs['input_ids'][0].cpu())
        del inputs
        output['gen_seq'].append(gen_seq)
        output['gen_toks'].append(gen_toks)
        output['gen_ids'].append(gen_ids)
        output['tok_idx'].append(tok_idx)
        output['gen_toks_trunc'].append(gen_toks_trunc)
        output['gen_seq_trunc'].append(gen_seq_trunc)
        output['gen_ids_trunc'].append(gen_ids_trunc)
        output['tok_idx_trunc'].append(tok_idx_trunc)
        output['truncated_original'].append(truncated_original)
        output['tok_idx_trunc_wo_nl'].append(tok_idx_trunc_wo_nl)
        output['word_ids_list_wo_nl'].append(word_ids_list_wo_nl)
        output['gen_toks_trunc_wo_nl'].append(gen_toks_trunc_wo_nl)
        output['removed_newlines'].append(removed_newlines)
        output['gen_seq_trunc_split_wl'].append(gen_seq_trunc_split_wl)
        output['gen_ids_trunc_wo_nl'].append(gen_ids_trunc_wo_nl)
        output['tok_idx_trunc_wo_nl_wo_punct'].append(tok_idx_trunc_wo_nl_wo_punct)
        output['word_ids_list_wo_nl_wo_punct'].append(word_ids_list_wo_nl_wo_punct)
        output['gen_toks_trunc_wo_nl_wo_punct'].append(gen_toks_trunc_wo_nl_wo_punct)
        output['gen_ids_trunc_wo_nl_wo_punct'].append(gen_ids_trunc_wo_nl_wo_punct)
        output['alignment_mismatch'].append(alignment_mismatch)
        output['remove_ctr'].append(remove_ctr)
        output['cut_nl_idx'].append(cut_nl_idx)


    # save config
    path_to_save = os.path.join(tensor_path, model_name)
    config_filename = f'{model_name}_{decoding_strategy}_model-config.json'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    with open(os.path.join(path_to_save, config_filename), 'w') as f:
        json.dump(model.config.to_dict(), f)
    # save output
    output_df = pd.DataFrame(output)
    del output
    out_path = os.path.join(seq_path, model_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    first_item_id, last_item_id = '', ''
    if len(item_ids[0]) == 6:
        first_item_id = item_ids[0][-2:]
    elif len(item_ids[0]) == 7:
        first_item_id = item_ids[0][-3:]
    if len(item_ids[-1]) == 6:
        last_item_id = item_ids[-1][-2:]
    elif len(item_ids[-1]) == 7:
        last_item_id = item_ids[-1][-3:]
    filename = f'{model_name}_decoding-{decoding_strategy}_' \
               f'{args.max_new_tokens}_items-{first_item_id}-{last_item_id}.csv'
    output_df.to_csv(os.path.join(out_path, filename), index=True)