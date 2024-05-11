#!/usr/bin/env python3
"""
Contains helper functions for the analyses.
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore
from typing import Dict, Any

from transformers import BertTokenizerFast, BertConfig
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from typing import Optional, Dict, Any, List

from transformers.models.bert.modeling_bert import BertEncoder, BertPooler


def compute_word_length(arr):
    arr = arr.astype('float64')
    arr[arr == 0] = 1 / ( 0 + 0.5)
    arr[arr != 0] = 1 / (arr[arr != 0])
    return arr


def pad_seq(seqs, max_len, pad_value, dtype=np.compat.long):
    padded = np.full((len(seqs), max_len), fill_value=pad_value, dtype=dtype)
    for i, seq in enumerate(seqs):
        padded[i, 0] = 0
        padded[i, 1:len(seq) + 1] = seq
        if pad_value != 0:
            padded[i, len(seq) + 1:] = pad_value - 1
    return padded


def pad_seq_with_nan(seqs, max_len, dtype=np.compat.long):
    padded = np.full((len(seqs), max_len), fill_value=np.nan, dtype=dtype)
    for i, seq in enumerate(seqs):
        padded[i, 1:(len(seq) + 1)] = seq
    return padded


def prepare_eyettention_input(
        path_to_fixations: str,
        path_to_ratings: str,
        path_to_reading_measures: str,
        tokenizer: BertTokenizerFast,
        max_sn_token: int,
        max_sp_token: int,
        max_sn_len: int,
        max_sp_len: int,
):
    """
    Prepare the input for the Eyettention model.
    :param path_to_fixations: path to the fixations data
    :param path_to_ratings: path to the ratings data
    :param path_to_reading_measures: path to the reading measures data
    :param tokenizer: the tokenizer to use
    :param max_sn_token: the maximum number of tokens/subwords in a text
    :param max_sp_token: the maximum number of tokens/subwords in a scanpath
    :param max_sn_len: the maximum number of words in a text
    :param max_sp_len: the maximum number of words in a scanpath
    :param normalize_ratings: whether to normalize the ratings
    """
    fixations = pd.read_csv(path_to_fixations, sep='\t')
    ratings = pd.read_csv(path_to_ratings, sep='\t')
    reading_measures = pd.read_csv(path_to_reading_measures, sep='\t')

    # group fixations by subject id and item id and add group dfs to a list
    fixations_dfs = list()
    grouped_fixations = fixations.groupby(['subject_id', 'item_id'])
    for name, group in grouped_fixations:
        fixations_dfs.append(group)

    # group ratings by subject id, z-score transform the difficulty and engaging ratings and add groups to dict
    ratings_dfs = dict()
    grouped_ratings = ratings.groupby('subject_id')
    for name, group in grouped_ratings:
        group['RATING_DIFFICULTY_VALUE_zscore'] = zscore(group['RATING_DIFFICULTY_VALUE'])
        group['RATING_ENGAGING_VALUE_zscore'] = zscore(group['RATING_ENGAGING_VALUE'])
        ratings_dfs[name] = group

    # TODO remove break index and break statements below
    break_idx = None

    # save all rating values in lists
    ratings_difficulty, ratings_difficulty_zscore = list(), list()
    ratings_engaging, ratings_engaging_zscore = list(), list()
    for fix_df_idx, fix_df in enumerate(fixations_dfs):

        if break_idx is not None:
            if fix_df_idx == break_idx:
                break

        subject_id = fix_df['subject_id'].unique().item()
        item_id = fix_df['item_id'].unique().item()
        difficulty = ratings_dfs[subject_id].loc[
            ratings_dfs[subject_id]['item_id'] == item_id
            ]['RATING_DIFFICULTY_VALUE'].item()
        difficulty_zscore = ratings_dfs[subject_id].loc[
            ratings_dfs[subject_id]['item_id'] == item_id
            ]['RATING_DIFFICULTY_VALUE_zscore'].item()
        engaging = ratings_dfs[subject_id].loc[
            ratings_dfs[subject_id]['item_id'] == item_id
            ]['RATING_ENGAGING_VALUE'].item()
        engaging_zscore = ratings_dfs[subject_id].loc[
            ratings_dfs[subject_id]['item_id'] == item_id
            ]['RATING_ENGAGING_VALUE_zscore'].item()
        ratings_difficulty.append(difficulty)
        ratings_difficulty_zscore.append(difficulty_zscore)
        ratings_engaging.append(engaging)
        ratings_engaging_zscore.append(engaging_zscore)

    SN_input_ids, SN_attention_mask, SN_WORD_len, WORD_ids_sn = list(), list(), list(), list()
    SP_input_ids, SP_attention_mask, WORD_ids_sp = list(), list(), list()
    SP_ordinal_pos, SP_fix_dur = list(), list()
    SP_landing_pos = list()
    SP_len = list()

    subject_ids = list()

    text_types = ['summarization', 'non-fiction', 'poetry', 'words_given', 'article_synopsis', 'fiction']
    text_type_labels = list()
    label_to_index = {label: i for i, label in enumerate(text_types)}

    # temp_max_sn_len, temp_max_sn_token = 0, 0
    # temp_max_sp_len, temp_max_sp_token = 0, 0
    # all_sn_lens, all_sn_tokens = list(), list()
    # all_sp_lens, all_sp_tokens = list(), list()

    for fix_df_idx, fix_df in enumerate(fixations_dfs):

        if break_idx is not None:
            if fix_df_idx == break_idx:
                break

        print(f'--- preparing scanpath {fix_df_idx + 1}/{len(fixations_dfs)} ---')

        item_id = fix_df['item_id'].unique().item()
        model = fix_df['model'].unique().item()
        decoding_strategy = fix_df['decoding_strategy'].unique().item()
        subject_id = fix_df['subject_id'].unique().item()

        subject_ids.append(subject_id)

        # locate the corresponding reading measures in the reading measures data frame to get the sentence and word lens
        rms_df = reading_measures.loc[
            (reading_measures['item_id'] == item_id)
            & (reading_measures['model'] == model)
            & (reading_measures['decoding_strategy'] == decoding_strategy)
            & (reading_measures['subject_id'] == subject_id)
            ]

        # get the text type
        text_type = rms_df['task'].unique().item()
        text_type_labels.append(label_to_index[text_type])

        # get the sentence and word lens
        sn = rms_df['word'].tolist()
        sn_len = (len(sn))
        sn_word_len = rms_df['word_length_without_punct'].values
        sn_word_len = compute_word_length(sn_word_len)
        sn_str = '[CLS] ' + ' '.join(sn) + ' [SEP]'

        # cur_max_sn_len = len(sn_str.split())
        # temp_max_sn_len = max(temp_max_sn_len, cur_max_sn_len)
        # all_sn_lens.append(len(sn_str.split()))

        # tokenize the sentence
        tokens = tokenizer.encode_plus(
            sn_str.split(),
            add_special_tokens=False,
            truncation=False,
            max_length=max_sn_token,
            padding='max_length',
            return_attention_mask=True,
            is_split_into_words=True,
        )
        encoded_sn = tokens['input_ids']
        mask_sn = tokens['attention_mask']
        word_ids_sn = tokens.word_ids()
        word_ids_sn = [val if val is not None else np.nan for val in word_ids_sn]

        # cur_max_sn_token = len(encoded_sn)
        # temp_max_sn_token = max(temp_max_sn_token, cur_max_sn_token)
        # all_sn_tokens.append(len(encoded_sn))

        # get the fixation indices
        # in the data, the word id of the first word is 0 but here 0 will refer to the CLS token
        # so add 1 to all fixation indices
        sp_word_pos = np.array(fix_df['word_id'].tolist()) + 1
        sp_fix_dur = fix_df['fixation_duration'].values
        sp_words = fix_df['word'].values

        # check for outliers, whether the recorded fixation durations are within reasonable limits
        # less than 50ms: attempt to merge with neighbouring fixation if fixation is on the same word, else delete
        outlier_idx = np.where(sp_fix_dur < 50)[0]
        if outlier_idx.size > 0:
            for out_idx in range(len(outlier_idx)):
                outlier_i = outlier_idx[out_idx]
                merge_flag = False

                # outliers are commonly found in the fixation of the last record and the first record, remove directly
                if outlier_i == len(sp_fix_dur)-1 or outlier_i == 0:
                    merge_flag = True
                else:
                    if outlier_i - 1 >= 0 and merge_flag == False:
                        # try to merge with the left fixation
                        if sp_words[outlier_i] == sp_words[outlier_i - 1]:
                            sp_fix_dur[outlier_i - 1] = sp_fix_dur[outlier_i - 1] + sp_fix_dur[outlier_i]
                            merge_flag = True

                    if outlier_i + 1 < len(sp_fix_dur) and merge_flag == False:
                        # try to merge with the right fixation
                        if sp_words[outlier_i] == sp_words[outlier_i + 1]:
                            sp_fix_dur[outlier_i + 1] = sp_fix_dur[outlier_i + 1] + sp_fix_dur[outlier_i]
                            merge_flag = True

                sp_word_pos = np.delete(sp_word_pos, outlier_i)
                sp_fix_dur = np.delete(sp_fix_dur, outlier_i)
                sp_words = np.delete(sp_words, outlier_i)
                outlier_idx = outlier_idx - 1

        sp_ordinal_pos = sp_word_pos.astype(int)
        SP_ordinal_pos.append(sp_ordinal_pos)
        SP_fix_dur.append(sp_fix_dur)

        sp_token_str = '[CLS] ' + ' '.join(sp_words) + ' [SEP]'

        # cur_max_sp_len = len(sp_token_str.split())
        # temp_max_sp_len = max(temp_max_sp_len, cur_max_sp_len)

        # tokenization and padding of the scanpath, i.e. the fixated word sequence
        sp_tokens = tokenizer.encode_plus(
            sp_token_str.split(),
            add_special_tokens=False,
            truncation=False,
            max_length=max_sp_token,
            padding='max_length',
            return_attention_mask=True,
            is_split_into_words=True,
        )

        encoded_sp = sp_tokens['input_ids']
        mask_sp = sp_tokens['attention_mask']
        # index starts from 0: CLS = 0 and SEP = last index
        word_ids_sp = sp_tokens.word_ids()
        word_ids_sp = [val if val is not None else np.nan for val in word_ids_sp]
        SP_input_ids.append(encoded_sp)
        SP_attention_mask.append(mask_sp)
        WORD_ids_sp.append(word_ids_sp)
        SP_len.append(len(sp_token_str.split()))

        # cur_max_sp_token = len(encoded_sp)
        # temp_max_sp_token = max(temp_max_sp_token, cur_max_sp_token)
        # all_sp_lens.append(len(sp_token_str.split()))
        # all_sp_tokens.append(len(encoded_sp))

        # sentence information
        SN_input_ids.append(encoded_sn)
        SN_attention_mask.append(mask_sn)
        SN_WORD_len.append(sn_word_len)
        WORD_ids_sn.append(word_ids_sn)

    # padding for batch computation
    SP_ordinal_pos = pad_seq(seqs=SP_ordinal_pos, max_len=max_sp_len, pad_value=max_sn_len)
    SP_fix_dur = pad_seq(seqs=SP_fix_dur, max_len=max_sp_len, pad_value=0)
    SN_WORD_len = pad_seq_with_nan(SN_WORD_len, max_sn_len, dtype=np.float32)

    # assign type
    SN_input_ids = np.asarray(SN_input_ids, dtype=np.int64)
    SN_attention_mask = np.asarray(SN_attention_mask, dtype=np.float32)
    SP_input_ids = np.asarray(SP_input_ids, dtype=np.int64)
    SP_attention_mask = np.asarray(SP_attention_mask, dtype=np.float32)
    WORD_ids_sn = np.asarray(WORD_ids_sn)
    WORD_ids_sp = np.asarray(WORD_ids_sp)
    SP_len = np.asarray(SP_len)

    # convert the non-normalized ratings to one-hot encoded labels
    # the -1 ensures that indexing of the labels starts from 0
    ratings_difficulty_one_hot = nn.functional.one_hot(torch.tensor(ratings_difficulty) - 1, num_classes=5)
    ratings_engaging_one_hot = nn.functional.one_hot(torch.tensor(ratings_engaging) - 1, num_classes=5)

    # one-hot encode the text types
    text_types_one_hot = nn.functional.one_hot(torch.tensor(text_type_labels), num_classes=6)

    ratings_difficulty = np.asarray(ratings_difficulty, dtype=np.int64)
    ratings_difficulty_zscore = np.asarray(ratings_difficulty_zscore, dtype=np.float32)
    ratings_engaging = np.asarray(ratings_engaging, dtype=np.int64)
    ratings_engaging_zscore = np.asarray(ratings_engaging_zscore, dtype=np.float32)

    data = {
        'SN_input_ids': SN_input_ids,
        'SN_attention_mask': SN_attention_mask,
        'SN_WORD_len': SN_WORD_len,
        'WORD_ids_sn': WORD_ids_sn,
        'SP_input_ids': SP_input_ids,
        'SP_attention_mask': SP_attention_mask,
        'WORD_ids_sp': WORD_ids_sp,
        'SP_ordinal_pos': np.array(SP_ordinal_pos),
        'SP_fix_dur': np.array(SP_fix_dur),
        'SP_len': SP_len,
        'ratings_difficulty': ratings_difficulty,
        'ratings_difficulty_one_hot': ratings_difficulty_one_hot,
        'ratings_difficulty_zscore': ratings_difficulty_zscore,
        'ratings_engaging': ratings_engaging,
        'ratings_engaging_one_hot': ratings_engaging_one_hot,
        'ratings_engaging_zscore': ratings_engaging_zscore,
        'subject_ids': np.array(subject_ids),
        'text_type_labels': np.array(text_type_labels),
        'text_type_labels_one_hot': text_types_one_hot,
    }

    return data


class EMTeCDataset(Dataset):

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def __len__(self):
        return len(self.data['SN_input_ids'])

    def __getitem__(self, idx):
        sample = {}
        sample['sn_input_ids'] = self.data['SN_input_ids'][idx, :]
        sample['sn_attention_mask'] = self.data['SN_attention_mask'][idx, :]
        sample['sn_word_len'] = self.data['SN_WORD_len'][idx, :]
        sample['word_ids_sn'] = self.data['WORD_ids_sn'][idx, :]
        sample['sp_input_ids'] = self.data['SP_input_ids'][idx, :]
        sample['sp_attention_mask'] = self.data['SP_attention_mask'][idx, :]
        sample['word_ids_sp'] = self.data['WORD_ids_sp'][idx, :]
        sample['sp_pos'] = self.data['SP_ordinal_pos'][idx, :]
        sample['sp_fix_dur'] = self.data['SP_fix_dur'][idx, :]
        sample['sp_len'] = self.data['SP_len'][idx]
        sample['rating_difficulty'] = self.data['ratings_difficulty'][idx]
        sample['rating_difficulty_one_hot'] = self.data['ratings_difficulty_one_hot'][idx, :]
        sample['rating_difficulty_zscore'] = self.data['ratings_difficulty_zscore'][idx]
        sample['rating_engaging'] = self.data['ratings_engaging'][idx]
        sample['rating_engaging_one_hot'] = self.data['ratings_engaging_one_hot'][idx, :]
        sample['rating_engaging_zscore'] = self.data['ratings_engaging_zscore'][idx]
        sample['subject_id'] = self.data['subject_ids'][idx]
        sample['text_type'] = self.data['text_type_labels'][idx]
        sample['text_type_one_hot'] = self.data['text_type_labels_one_hot'][idx, :]
        return sample


def calculate_mean_std(dataloader, feat_key, padding_value=0, scale=1):
    # calculate mean
    total_sum = 0
    total_num = 0
    for batch in dataloader:
        batch.keys()
        feat = batch[feat_key] / scale
        feat = torch.nan_to_num(feat)
        total_num += len(feat.view(-1).nonzero())
        total_sum += feat.sum()
    feat_mean = total_sum / total_num
    # calculate std
    sum_of_squared_error = 0
    for batch in dataloader:
        batch.keys()
        feat = batch[feat_key] / scale
        feat = torch.nan_to_num(feat)
        mask = ~torch.eq(feat, padding_value)
        sum_of_squared_error += (((feat - feat_mean).pow(2)) * mask).sum()
    feat_std = torch.sqrt(sum_of_squared_error / total_num)
    return feat_mean, feat_std


# this loss doesn't make sense as it attributes higher weights to the higher ratings
# def ordinal_logistic_loss(
#         predictions: torch.Tensor,
#         targets: torch.Tensor,
# ):
#     """
#     Ordered logisitc loss function.
#     :param predictions: predicted logits or probabilities (tensor)
#     :param targets: true labels (tensor)
#     :return: ordinal logistic loss (tensor)
#     """
#     loss = 0
#     for i in range(predictions.size(0)):
#         # compute the cumulative logits for each class
#         breakpoint()
#         cumulative_logits = torch.cumsum(predictions[i], dim=0)
#         # compute the loss for each class
#         class_loss = F.cross_entropy(cumulative_logits.unsqueeze(0), targets[i])
#         loss += class_loss
#     return loss / predictions.size(0)


class OrdinalHingeLoss(nn.Module):
    def __init__(self, num_classes, device):
        super(OrdinalHingeLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device

        # thresholds are learnable parameters representing the thresholds that define the boundaries between classes.
        # they are initialized uniformly between -1 and 1 and are learned during training.
        self.thresholds = nn.Parameter(torch.Tensor(num_classes)).to(self.device)
        nn.init.uniform_(self.thresholds, -1, 1)

    def forward(self, predictions, targets):
        """
        Compute the ordinal hinge loss.
        :param predictions: Predicted logits (Tensor)
        :param targets: True labels (Tensor)
        :return: Ordinal hinge loss (Tensor)
        """
        # compute the differences between predicted logits and thresholds
        diff = predictions.unsqueeze(1) - self.thresholds.unsqueeze(0)
        # compute the hinge loss for each class
        hinge_loss = F.relu(diff * (targets.unsqueeze(1) - torch.arange(self.num_classes).to(self.device).unsqueeze(0)).float())
        # sum hinge loss over classes
        loss = hinge_loss.sum(dim=1)
        # average loss over samples
        return loss.mean()


def get_kfold(
        inputs: torch.Tensor,
        labels: torch.Tensor,
        split: str,
        n_splits: int,
        group: Optional[np.array] = None,
):
    """
    Returns a k-fold iterator depending on the split type (by subject or random).
    :param inputs: The inputs to the model
    :param labels: The labels
    :param split: Subject or random
    :param n_splits: How many folds for the k-fold CV
    :param group: if group k-fold, the group
    :return: k-fold iterator
    """
    if split == 'subject':
        kfold = GroupKFold(n_splits=n_splits)
        kfold.get_n_splits(inputs, labels, groups=group)
        return kfold.split(inputs, labels, groups=group)
    elif split == 'random':
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        kfold.get_n_splits(inputs, labels)
        return kfold.split(inputs, labels)
    else:
        raise NotImplementedError(f'The cross-validation split {split} is not implemented.')


def subset_data_kfold(
        data: Dict[str, Any],
        train_idx: np.array,
        test_idx: np.array,
):
    """
    Subsets the dataset according to the k-fold cross-validation indices.
    :param data: The data
    :param train_idx: the train indices
    :param test_idx: the test indices
    :return: Returns the data split into train and test data
    """
    train_data, test_data = dict(), dict()
    for key in data.keys():
        train_data[key] = data[key][train_idx]
        test_data[key] = data[key][test_idx]
    return train_data, test_data


def split_train_val_eyettention(
        train_data: Dict[str, Any],
        val_size: float = 0.1,
):
    """
        Split the train data into train and validation data.
        :param train_data:
        :param val_size:
        :return:
    """
    SN_input_ids_train, SN_input_ids_test, SN_attention_mask_train, SN_attention_mask_test, SN_WORD_len_train, SN_WORD_len_test, WORD_ids_sn_train, WORD_ids_sn_test, SP_input_ids_train, SP_input_ids_test, SP_attention_mask_train, SP_attention_mask_test, WORD_ids_sp_train, WORD_ids_sp_test, SP_ordinal_pos_train, SP_ordinal_pos_test, SP_fix_dur_train, SP_fix_dur_test, SP_len_train, SP_len_test, ratings_difficulty_train, ratings_difficulty_test, ratings_difficulty_one_hot_train, ratings_difficulty_one_hot_test, ratings_difficulty_zscore_train, ratings_difficulty_zscore_test, ratings_engaging_train, ratings_engaging_test, ratings_engaging_one_hot_train, ratings_engaging_one_hot_test, ratings_engaging_zscore_train, ratings_engaging_zscore_test, subject_ids_train, subject_ids_test, text_type_labels_train, text_type_labels_test, text_type_labels_one_hot_train, text_type_labels_one_hot_test = train_test_split(
        train_data['SN_input_ids'], train_data['SN_attention_mask'], train_data['SN_WORD_len'], train_data['WORD_ids_sn'], train_data['SP_input_ids'], train_data['SP_attention_mask'],
        train_data['WORD_ids_sp'], train_data['SP_ordinal_pos'], train_data['SP_fix_dur'], train_data['SP_len'], train_data['ratings_difficulty'], train_data['ratings_difficulty_one_hot'],
        train_data['ratings_difficulty_zscore'], train_data['ratings_engaging'], train_data['ratings_engaging_one_hot'], train_data['ratings_engaging_zscore'], train_data['subject_ids'],
        train_data['text_type_labels'], train_data['text_type_labels_one_hot'],
        test_size=val_size, random_state=0, shuffle=True,
    )
    new_train_data = {
        'SN_input_ids': SN_input_ids_train,
        'SN_attention_mask': SN_attention_mask_train,
        'SN_WORD_len': SN_WORD_len_train,
        'WORD_ids_sn': WORD_ids_sn_train,
        'SP_input_ids': SP_input_ids_train,
        'SP_attention_mask': SP_attention_mask_train,
        'WORD_ids_sp': WORD_ids_sp_train,
        'SP_ordinal_pos': SP_ordinal_pos_train,
        'SP_fix_dur': SP_fix_dur_train,
        'SP_len': SP_len_train,
        'ratings_difficulty': ratings_difficulty_train,
        'ratings_difficulty_one_hot': ratings_difficulty_one_hot_train,
        'ratings_difficulty_zscore': ratings_difficulty_zscore_train,
        'ratings_engaging': ratings_engaging_train,
        'ratings_engaging_one_hot': ratings_engaging_one_hot_train,
        'ratings_engaging_zscore': ratings_engaging_zscore_train,
        'subject_ids': subject_ids_train,
        'text_type_labels': text_type_labels_train,
        'text_type_labels_one_hot': text_type_labels_one_hot_train,
    }
    val_data = {
        'SN_input_ids': SN_input_ids_test,
        'SN_attention_mask': SN_attention_mask_test,
        'SN_WORD_len': SN_WORD_len_test,
        'WORD_ids_sn': WORD_ids_sn_test,
        'SP_input_ids': SP_input_ids_test,
        'SP_attention_mask': SP_attention_mask_test,
        'WORD_ids_sp': WORD_ids_sp_test,
        'SP_ordinal_pos': SP_ordinal_pos_test,
        'SP_fix_dur': SP_fix_dur_test,
        'SP_len': SP_len_test,
        'ratings_difficulty': ratings_difficulty_test,
        'ratings_difficulty_one_hot': ratings_difficulty_one_hot_test,
        'ratings_difficulty_zscore': ratings_difficulty_zscore_test,
        'ratings_engaging': ratings_engaging_test,
        'ratings_engaging_one_hot': ratings_engaging_one_hot_test,
        'ratings_engaging_zscore': ratings_engaging_zscore_test,
        'subject_ids': subject_ids_test,
        'text_type_labels': text_type_labels_test,
        'text_type_labels_one_hot': text_type_labels_one_hot_test,
    }
    return new_train_data, val_data


def gradient_clipping(model, clip: int = 10):
    nn.utils.clip_grad_norm_(model.parameters(), clip)


def prepare_kan_input(
        path_to_rms: str,
        path_to_ratings: str,
        exclude_subjects: List[str],
):
    features = [
        'FFD', 'SFD', 'FD', 'FPRT', 'FRT', 'TFT', 'RRT',
        'RPD_inc', 'RPD_exc', 'RBRT', 'Fix', 'FPF', 'RR', 'FPReg', 'TRC_out',
        'TRC_in', 'SL_in', 'SL_out', 'TFC', 'n_dep_left', 'n_dep_right', 'distance_to_head',
        'word_length_without_punct', 'word_freq', 'zipf_freq', 'surprisal_gpt2',
    ]
    data_dict = {
        'features': list(),
        'difficulty': list(),
        'engaging': list(),
        'subject_id': list(),
        'difficulty_zscore': list(),
        'engaging_zscore': list(),
    }
    rms = pd.read_csv(path_to_rms, sep='\t')
    rms_grouped = rms.groupby(['subject_id', 'item_id'])
    rms_list = list()
    for name, group in rms_grouped:
        rms_list.append(group)

    results = pd.read_csv(path_to_ratings, sep='\t')
    ratings_dfs = dict()
    grouped_ratings = results.groupby('subject_id')
    for name, group in grouped_ratings:
        if name in exclude_subjects:
            continue
        group['RATING_DIFFICULTY_VALUE_zscore'] = zscore(group['RATING_DIFFICULTY_VALUE'])
        group['RATING_ENGAGING_VALUE_zscore'] = zscore(group['RATING_ENGAGING_VALUE'])
        ratings_dfs[name] = group

    for text_df_idx, text_df in enumerate(rms_list):
        print(f'--- preparing text {text_df_idx}')
        subject_id = text_df['subject_id'].unique().item()
        item_id = text_df['item_id'].unique().item()
        if subject_id in exclude_subjects:
            continue
        feature_vector = list()
        for feature in features:
            feature_vector.append(text_df[feature].mean())
            feature_vector.append(text_df[feature].std())
            feature_vector.append(text_df[feature].median())
        difficulty = ratings_dfs[subject_id].loc[ratings_dfs[subject_id]['item_id'] == item_id][
            'RATING_DIFFICULTY_VALUE'].item()
        difficulty_zscore = ratings_dfs[subject_id].loc[ratings_dfs[subject_id]['item_id'] == item_id][
            'RATING_DIFFICULTY_VALUE_zscore'].item()
        engaging = ratings_dfs[subject_id].loc[ratings_dfs[subject_id]['item_id'] == item_id]['RATING_ENGAGING_VALUE'].item()
        engaging_zscore = ratings_dfs[subject_id].loc[ratings_dfs[subject_id]['item_id'] == item_id][
            'RATING_ENGAGING_VALUE_zscore'].item()
        data_dict['features'].append(feature_vector)
        data_dict['difficulty'].append(difficulty)
        data_dict['engaging'].append(engaging)
        data_dict['subject_id'].append(subject_id)
        data_dict['difficulty_zscore'].append(difficulty_zscore)
        data_dict['engaging_zscore'].append(engaging_zscore)

    # data_dict['features'] = torch.tensor(data_dict['features'])
    # data_dict['difficulty'] = torch.tensor(data_dict['difficulty'])
    # data_dict['engaging'] = torch.tensor(data_dict['engaging'])
    # data_dict['difficulty_zscore'] = torch.tensor(data_dict['difficulty_zscore'])
    # data_dict['engaging_zscore'] = torch.tensor(data_dict['engaging_zscore'])

    return data_dict


def kan_train_test_split(
        data: Dict[str, torch.tensor],
        label: str,
        device: str,
        task: str,
        test_size: float = 0.2,
):
    features_train, features_test, difficulty_train, difficulty_test, engaging_train, engaging_test, difficulty_zscore_train, difficulty_zscore_test, engaging_zscore_train, engaging_zscore_test = train_test_split(
        data['features'], data['difficulty'], data['engaging'], data['difficulty_zscore'], data['engaging_zscore'],
        test_size=test_size, random_state=0, shuffle=True,
    )
    kan_data = dict()

    # kan_data['train_input'] = torch.tensor(features_train).double()
    # kan_data['test_input'] = torch.tensor(features_test).double()
    #
    # if label == 'difficulty':
    #     kan_data['train_label'] = torch.tensor(difficulty_train).double()
    #     kan_data['test_label'] = torch.tensor(difficulty_test).double()
    # elif label == 'engaging':
    #     kan_data['train_label'] = torch.tensor(engaging_train).double()
    #     kan_data['test_label'] = torch.tensor(engaging_test).double()
    # elif label == 'difficulty_zscore':
    #     kan_data['train_label'] = torch.tensor(difficulty_zscore_train).double()
    #     kan_data['test_label'] = torch.tensor(difficulty_zscore_test).double()
    # elif label == 'engaging_zscore':
    #     kan_data['train_label'] = torch.tensor(engaging_zscore_train).double()
    #     kan_data['test_label'] = torch.tensor(engaging_zscore_test).double()
    # else:
    #     raise NotImplementedError

    if task == 'classification':
        dtype = torch.long
    else:
        dtype = torch.float32

    kan_data['train_input'] = torch.tensor(features_train).to(torch.float32).to(device)
    kan_data['test_input'] = torch.tensor(features_test).to(torch.float32).to(device)

    if label == 'difficulty':
        kan_data['train_label'] = (torch.tensor(difficulty_train) - 1).to(dtype).to(device)
        kan_data['test_label'] = (torch.tensor(difficulty_test) - 1).to(dtype).to(device)
    elif label == 'engaging':
        kan_data['train_label'] = (torch.tensor(engaging_train) - 1).to(dtype).to(device)
        kan_data['test_label'] = (torch.tensor(engaging_test) - 1).to(dtype).to(device)
    elif label == 'difficulty_zscore':
        kan_data['train_label'] = torch.tensor(difficulty_zscore_train).to(torch.float32).to(device)
        kan_data['test_label'] = torch.tensor(difficulty_zscore_test).to(torch.float32).to(device)
    elif label == 'engaging_zscore':
        kan_data['train_label'] = torch.tensor(engaging_zscore_train).to(torch.float32).to(device)
        kan_data['test_label'] = torch.tensor(engaging_zscore_test).to(torch.float32).to(device)
    else:
        raise NotImplementedError

    return kan_data


def prepare_bert_input(
        path_to_rms: str,
        path_to_ratings: str,
        exclude_subjects: List[str],
        max_sn_len: int,
):
    """

    :param path_to_rms:
    :param path_to_ratings:
    :param exclude_subjects:
    :return:
    """
    text_types = ['summarization', 'non-fiction', 'poetry', 'words_given', 'article_synopsis', 'fiction']
    label_to_index = {label: i for i, label in enumerate(text_types)}

    feature_names = [
        'FFD', 'SFD', 'FD', 'FPRT', 'FRT', 'TFT', 'RRT',
        'RPD_inc', 'RPD_exc', 'RBRT', 'Fix', 'FPF', 'RR', 'FPReg', 'TRC_out',
        'TRC_in', 'SL_in', 'SL_out', 'TFC', 'n_dep_left', 'n_dep_right', 'distance_to_head',
        'word_length_without_punct', 'word_freq', 'zipf_freq', 'surprisal_gpt2',
    ]
    feature_names = [
        'FFD', 'SFD', 'FD', 'FPRT', 'FRT', 'TFT', 'RRT',
        'RPD_inc', 'RPD_exc', 'RBRT',
        #'Fix', 'FPF',
        'RR', 'FPReg', 'TRC_out',
        'TRC_in', 'SL_in', 'SL_out', 'TFC', 'n_dep_left', 'n_dep_right', 'distance_to_head',
        'word_length_without_punct', 'word_freq', 'zipf_freq', 'surprisal_gpt2',
    ]
    data_dict = {
        'features': list(),
        'difficulty': list(),
        'engaging': list(),
        'subject_id': list(),
        'difficulty_zscore': list(),
        'engaging_zscore': list(),
        'subject_ids': list(),
        'text_type': list(),
    }

    features = list()
    difficulty_labels = list()
    difficulty_zscore_labels = list()
    engaging_labels = list()
    engaging_zscore_labels = list()
    subject_ids = list()
    text_types_int = list()
    text_types_str = list()

    rms = pd.read_csv(path_to_rms, sep='\t')
    rms_grouped = rms.groupby(['subject_id', 'item_id'])
    rms_list = list()
    for name, group in rms_grouped:
        rms_list.append(group)

    results = pd.read_csv(path_to_ratings, sep='\t')
    ratings_dfs = dict()
    grouped_ratings = results.groupby('subject_id')
    for name, group in grouped_ratings:
        if name in exclude_subjects:
            continue
        group['RATING_DIFFICULTY_VALUE_zscore'] = zscore(group['RATING_DIFFICULTY_VALUE'])
        group['RATING_ENGAGING_VALUE_zscore'] = zscore(group['RATING_ENGAGING_VALUE'])
        ratings_dfs[name] = group

    for text_df_idx, text_df in enumerate(rms_list):
        print(f'--- preparing text {text_df_idx}')
        subject_id = text_df['subject_id'].unique().item()
        item_id = text_df['item_id'].unique().item()
        text_type_str = text_df['task'].unique().item()
        text_type_int = label_to_index[text_type_str]
        if subject_id in exclude_subjects:
            continue

        # list to hold the individual word-level lists
        feature_matrix = list()
        for idx, row in text_df.iterrows():
            # list that holds the word-level values
            feature_vector = list()
            for feature in feature_names:
                feature_vector.append(row[feature])
            feature_matrix.append(feature_vector)

        difficulty = ratings_dfs[subject_id].loc[ratings_dfs[subject_id]['item_id'] == item_id][
            'RATING_DIFFICULTY_VALUE'].item()
        difficulty_zscore = ratings_dfs[subject_id].loc[ratings_dfs[subject_id]['item_id'] == item_id][
            'RATING_DIFFICULTY_VALUE_zscore'].item()
        engaging = ratings_dfs[subject_id].loc[ratings_dfs[subject_id]['item_id'] == item_id][
            'RATING_ENGAGING_VALUE'].item()
        engaging_zscore = ratings_dfs[subject_id].loc[ratings_dfs[subject_id]['item_id'] == item_id][
            'RATING_ENGAGING_VALUE_zscore'].item()

        features.append(feature_matrix)
        difficulty_labels.append(difficulty)
        difficulty_zscore_labels.append(difficulty_zscore)
        engaging_labels.append(engaging)
        engaging_zscore_labels.append(engaging_zscore)
        subject_ids.append(subject_id)
        text_types_str.append(text_type_str)
        text_types_int.append(text_type_int)


    # one-hot encoding
    difficulty_onehot_labels = nn.functional.one_hot(torch.tensor(difficulty_labels) - 1, num_classes=5)
    engaging_onehot_labels = nn.functional.one_hot(torch.tensor(engaging_labels) - 1, num_classes=5)
    text_types_onehot = nn.functional.one_hot(torch.tensor(text_types_int), num_classes=6)

    # pad the features
    features, mask = pad_bert_features_input(
        features=features,
        max_sn_len=max_sn_len,
    )

    difficulty_labels = np.asarray(difficulty_labels, dtype=np.int64) - 1
    difficulty_zscore_labels = np.asarray(difficulty_zscore_labels, dtype=np.float32)
    engaging_labels = np.asarray(engaging_labels, dtype=np.int64) - 1
    engaging_zscore_labels = np.asarray(engaging_zscore_labels, dtype=np.float32)
    subject_ids = np.array(subject_ids)
    text_types_str = np.array(text_types_str)
    text_types_int = np.asarray(text_types_int, dtype=np.int64)

    data = {
        'features': features,
        'mask': mask,
        'difficulty_labels': difficulty_labels,
        'difficulty_zscore_labels': difficulty_zscore_labels,
        'difficulty_onehot_labels': difficulty_onehot_labels,
        'engaging_labels': engaging_labels,
        'engaging_zscore_labels': engaging_zscore_labels,
        'engaging_onehot_labels': engaging_onehot_labels,
        'subject_ids': subject_ids,
        'text_types_str': text_types_str,
        'text_types_int': text_types_int,
        'text_types_onehot': text_types_onehot,
    }

    return data


class EMTeCBert(Dataset):

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def __len__(self):
        return len(self.data['features'])

    def __getitem__(self, idx):
        sample = {}
        sample['features'] = self.data['features'][idx, :]
        sample['mask'] = self.data['mask'][idx, :]
        sample['difficulty_labels'] = self.data['difficulty_labels'][idx]
        sample['difficulty_zscore_labels'] = self.data['difficulty_zscore_labels'][idx]
        sample['difficulty_onehot_labels'] = self.data['difficulty_onehot_labels'][idx, :]
        sample['engaging_labels'] = self.data['engaging_labels'][idx]
        sample['engaging_zscore_labels'] = self.data['engaging_zscore_labels'][idx]
        sample['engaging_onehot_labels'] = self.data['engaging_onehot_labels'][idx, :]
        sample['subject_ids'] = self.data['subject_ids'][idx]
        sample['text_types_str'] = self.data['text_types_str'][idx]
        sample['text_types_int'] = self.data['text_types_int'][idx]
        sample['text_types_onehot'] = self.data['text_types_onehot'][idx, :]
        return sample


def pad_bert_features_input(
        features: List[list[int]],
        max_sn_len: int,
):
    features_tensors = torch.tensor
    mask = torch.tensor
    for idx, feature_matrix in enumerate(features):

        feature_matrix = torch.tensor(feature_matrix)
        feature_len = feature_matrix.shape[0]
        pad_len = max_sn_len - feature_len

        # create attention mask
        mask_ones = torch.ones(1, feature_len)
        mask_zeros = torch.zeros(1, pad_len)
        mask_complete = torch.cat((mask_ones, mask_zeros), dim=1)

        # create padding
        padding_shape = (pad_len, feature_matrix.shape[1])
        padding = torch.full(padding_shape, -1)

        # concatenate features and padding
        padded_features = torch.cat((feature_matrix, padding), dim=0)

        if idx == 0:
            features_tensors = padded_features
            mask = mask_complete.squeeze()
        elif idx == 1:
            features_tensors = torch.stack((features_tensors, padded_features), dim=0)
            mask = torch.stack((mask, mask_complete.squeeze()), dim=0)
        else:
            features_tensors = torch.cat((features_tensors, padded_features.unsqueeze(0)), dim=0)
            mask = torch.cat((mask, mask_complete), dim=0)


    return features_tensors, mask


class TransformerModel(nn.Module):

    def __init__(
            self,
            num_classes: int,
    ):
        super().__init__()
        self.num_classes = num_classes

        bert_config = BertConfig.from_pretrained('bert-base-cased')
        # we have 24 input features
        bert_config.hidden_size = 24

        self.bert = BertEncoder(bert_config)
        self.pooler = BertPooler(bert_config)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.linear = nn.Linear(bert_config.hidden_size, self.num_classes)

    def forward(self, features, attention_mask):


        attention_mask = self._prepare_attention_mask(attention_mask=attention_mask)
        output = self.bert(hidden_states=features, attention_mask=attention_mask).last_hidden_state
        output = self.pooler(output)
        output = self.dropout(output)
        output = self.linear(output)

        return output

    def _prepare_attention_mask(self, attention_mask):
        # the input attention mask is of shape [batch size, hidden]
        extended_attention_mask = attention_mask[:, None, None, :]
        # since the attentoin mask is 1 for positions we want to attend to and 0 for maked positions,
        # this operation will create a tensor which is 0 for positions we want to attend and the dtype's
        # smallest value for masked positions.
        # since we are adding it to the raw scores before the softmax, this is the same as
        # removing these entirely.
        dtype = torch.float32
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask


def split_train_val_bert(
        train_data: Dict[str, Any],
        val_size: float = 0.1,
):
    """

    :param train_data:
    :param val_size:
    :return:
    """
    features_train, features_test, mask_train, mask_test, difficulty_labels_train, difficulty_labels_test, difficulty_zscore_labels_train, difficulty_zscore_labels_test, difficulty_onehot_labels_train, difficulty_onehot_labels_test, engaging_labels_train, engaging_labels_test, engaging_zscore_labels_train, engaging_zscore_labels_test, engaging_onehot_labels_train, engaging_onehot_labels_test, subject_ids_train, subject_ids_test, text_types_str_train, text_types_str_test, text_types_int_train, text_types_int_test, text_types_onehot_train, text_types_onehot_test = train_test_split(
        train_data['features'], train_data['mask'], train_data['difficulty_labels'], train_data['difficulty_zscore_labels'],
        train_data['difficulty_onehot_labels'], train_data['engaging_labels'], train_data['engaging_zscore_labels'],
        train_data['engaging_onehot_labels'], train_data['subject_ids'], train_data['text_types_str'],
        train_data['text_types_int'], train_data['text_types_onehot'],
        test_size=val_size, random_state=0, shuffle=True,
    )
    new_train_data = {
        'features': features_train,
        'mask': mask_train,
        'difficulty_labels': difficulty_labels_train,
        'difficulty_zscore_labels': difficulty_zscore_labels_train,
        'difficulty_onehot_labels': difficulty_onehot_labels_train,
        'engaging_labels': engaging_labels_train,
        'engaging_zscore_labels': engaging_zscore_labels_train,
        'engaging_onehot_labels': engaging_onehot_labels_train,
        'subject_ids': subject_ids_train,
        'text_types_str': text_types_str_train,
        'text_types_int': text_types_int_train,
        'text_types_onehot': text_types_onehot_train,
    }
    val_data = {
        'features': features_test,
        'mask': mask_test,
        'difficulty_labels': difficulty_labels_test,
        'difficulty_zscore_labels': difficulty_zscore_labels_test,
        'difficulty_onehot_labels': difficulty_onehot_labels_test,
        'engaging_labels': engaging_labels_test,
        'engaging_zscore_labels': engaging_zscore_labels_test,
        'engaging_onehot_labels': engaging_onehot_labels_test,
        'subject_ids': subject_ids_test,
        'text_types_str': text_types_str_test,
        'text_types_int': text_types_int_test,
        'text_types_onehot': text_types_onehot_test,
    }
    return new_train_data, val_data





