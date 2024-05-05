#!/usr/bin/env python3
"""
Train the Eyettention architecture on a classification task of subjective text difficulty.
"""

from analyses.utils.utils_analyses import (
    prepare_eyettention_input,
    EMTeCDataset,
    calculate_mean_std,
    OrdinalHingeLoss,
    get_kfold,
    subset_data_kfold,
    split_train_val,
    gradient_clipping,
)
from transformers import BertTokenizerFast
from argparse import ArgumentParser
from analyses.utils.eyettention_model import Eyettention, ClassificationModel
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import pandas as pd
from collections import deque



def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--atten-type',
        type=str,
        default='global',
        help='The kind of attention window to use for the cross-attention in Eyettention.',
        choices=['global', 'local', 'local-g'],
    )
    parser.add_argument(
        '--eyettention-output',
        type=str,
        default='both_concat',
        choices=['last_hidden', 'context', 'both_concat'],
        help='Which Eyettention output to use as input to the classification',
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='The GPU to use for training.',
    )
    parser.add_argument(
        '--classification-input',
        type=str,
        default='last',
        choices=['last', 'all'],
        help='Whether to use all states or only the last state as input to the classification layers.',
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='cross-entropy',
        choices=['cross-entropy', 'ordinal-hinge', 'mse'],
        help='The loss function to use',
    )
    parser.add_argument(
        '--task',
        type=str,
        default='classification',
        choices=['classification', 'regression'],
        help='Whether to perform regression or classification.',
    )
    parser.add_argument(
        '--target',
        type=str,
        default='difficulty',
        choices=['difficulty', 'engaging'],
        help='Whether we try to predict subjective text difficulty or engagement of the text.',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='subject',
        choices=['subject', 'random'],
        help='How to split the data for cross-validation.',
    )
    return parser


def main():

    args = get_parser().parse_args()

    gpu = args.gpu
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_default_tensor_type('torch.FloatTensor')
    if torch.cuda.is_available():
        device = f'cuda:{gpu}'
    else:
        device = 'cpu'

    path_to_fixations = 'data/fixations_corrected.csv'
    path_to_ratings = 'data/participant_info/participant_results.csv'
    path_to_reading_measures = 'data/reading_measures_corrected.csv'
    path_to_participant_info = 'data/participant_info/participant_info.csv'

    # make sure the input arguments make sense
    if args.task == 'classification':
        n_targets = 5
        if args.loss == 'mse':
            raise RuntimeError('For classification, a categorical loss function should be used.')
    elif args.task == 'regression':
        n_targets = 1
        if args.loss != 'mse':
            raise RuntimeError('For regression, MSE should be used.')
    else:
        raise NotImplementedError(f'The task {args.task} is not implemented.')

    # config for training
    cf = {
        'model_pretrained': 'bert-base-cased',
        'lr': 1e-3,
        'max_grad_norm': 10,
        'n_epochs': 1000,
        'n_folds': 5,
        'atten_type': args.atten_type,
        'batch_size': 8,  # TODO: change
        'max_sn_len': 132,
        'max_sn_token': 169,
        'max_sp_len': 303,
        'max_sp_token': 463,
        'norm_type': 'z-score',
        'earlystop_patience': 20,
        'eyettention_output': args.eyettention_output,
        'classification_input': args.classification_input,
        'hidden_size': 128,
        'num_lstm_layers': 4,
        'loss': args.loss,
        'n_targets': n_targets,
        'target': args.target,
        'task': args.task,
        'split': args.split,
    }

    # # get a list of subject ids
    # participant_info = pd.read_csv(path_to_participant_info, sep='\t')
    # subject_ids = participant_info['subject_id'].tolist()

    tokenizer = BertTokenizerFast.from_pretrained(cf['model_pretrained'])

    data = prepare_eyettention_input(
        path_to_fixations=path_to_fixations,
        path_to_ratings=path_to_ratings,
        path_to_reading_measures=path_to_reading_measures,
        tokenizer=tokenizer,
        max_sn_token=cf['max_sn_token'],
        max_sp_token=cf['max_sp_token'],
        max_sn_len=cf['max_sn_len'],
        max_sp_len=cf['max_sp_len'],
    )



    # iterate through the folds for k-fold cross-validation
    for train_idx, test_idx in get_kfold(
        inputs=data['SN_input_ids'],
        labels=data['ratings_difficulty'],
        split=cf['split'],
        n_splits=cf['n_folds'],
        group=data['subject_ids'],
    ):
        # dict to hold the training progress
        loss_dict = {
            'train_loss': list(),
            'val_loss': list(),
            'test_ll': list(),
            'test_AUC': list(),
        }

        # split the data into train and test fold
        train_data, test_data = subset_data_kfold(data=data, train_idx=train_idx, test_idx=test_idx)

        # split the train data into train and validation data
        train_data, val_data = split_train_val(train_data=train_data)

        # wrap in dataset class
        train_dataset = EMTeCDataset(train_data)
        test_dataset = EMTeCDataset(test_data)
        val_dataset = EMTeCDataset(val_data)

        # get dataloader iterator
        train_dataloader = DataLoader(train_dataset, batch_size=cf['batch_size'], shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=cf['batch_size'], shuffle=False, drop_last=False)
        val_dataloader = DataLoader(val_dataset, batch_size=cf['batch_size'], shuffle=False, drop_last=True)

        # z-score normalization for gaze features (use the train data)
        fix_dur_mean, fix_dur_std = calculate_mean_std(
            dataloader=train_dataloader,
            feat_key='sp_fix_dur',
            padding_value=0,
            scale=1000,
        )
        sn_word_len_mean, sn_word_len_std = calculate_mean_std(
            dataloader=train_dataloader,
            feat_key='sn_word_len',
        )

        # load the model, the optimizer, and the loss function

        model = ClassificationModel(
            cf=cf,
            device=device,
        )
        model.train()
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cf['lr'])

        if cf['loss'] == 'cross-entropy':
            loss_fn = nn.CrossEntropyLoss(reduction='mean')
        elif cf['loss'] == 'ordinal-hinge':
            loss_fn = OrdinalHingeLoss(num_classes=cf['n_targets'], device=device)
        elif cf['loss'] == 'mse':
            loss_fn = nn.MSELoss()

        # variables for early stopping
        old_score = 1e10
        av_score = deque(maxlen=100)
        save_ep_counter = 0

        for epoch in range(cf['n_epochs']):
            model.train()

            for batch_idx, batch in enumerate(train_dataloader):

                print(f'--- training epoch {epoch} batch {batch_idx}', end='\t')

                sn_input_ids = batch['sn_input_ids'].to(device)
                sn_attention_mask = batch['sn_attention_mask'].to(device)
                word_ids_sn = batch['word_ids_sn'].to(device)
                sn_word_len = batch['sn_word_len'].to(device)

                sp_input_ids = batch['sp_input_ids'].to(device)
                sp_attention_mask = batch['sp_attention_mask'].to(device)
                word_ids_sp = batch['word_ids_sp'].to(device)
                sp_len = batch['sp_len'].to(device)

                sp_pos = batch['sp_pos'].to(device)
                sp_fix_dur = (batch['sp_fix_dur'] / 1000).to(device)

                # normalize gaze features
                mask = ~torch.eq(sp_fix_dur, 0)
                sp_fix_dur = (sp_fix_dur - fix_dur_mean) / fix_dur_std * mask
                sp_fix_dur = torch.nan_to_num(sp_fix_dur)
                sn_word_len = (sn_word_len - sn_word_len_mean) / sn_word_len_std
                sn_word_len = torch.nan_to_num(sn_word_len)

                if cf['target'] == 'difficulty':
                    if cf['task'] == 'classification':
                        labels = batch['rating_difficulty_one_hot'].to(device)
                    elif cf['task'] == 'regression':
                        labels = batch['rating_difficulty'].to(device)
                    else:
                        raise NotImplementedError
                elif cf['target'] == 'engaging':
                    if cf['task'] == 'classification':
                        labels = batch['rating_engaging_one_hot'].to(device)
                    elif cf['task'] == 'regression':
                        labels = batch['rating_engaging'].to(device)
                    else:
                        raise NotImplementedError

                # zero old gradients
                optimizer.zero_grad()

                # predict
                out = model(
                    sn_input_ids=sn_input_ids,
                    sn_attention_mask=sn_attention_mask,
                    sp_input_ids=sp_input_ids,
                    sp_pos=sp_pos,
                    word_ids_sn=word_ids_sn,
                    word_ids_sp=word_ids_sp,
                    sp_fix_dur=sp_fix_dur,
                    sn_word_len=sn_word_len,
                    sp_len=sp_len,
                )

                if args.loss == 'cross-entropy':
                    loss = loss_fn(out, labels.float())
                elif args.loss == 'ordinal-hinge':
                    loss = loss_fn(out, labels)
                elif args.loss == 'mse':
                    loss = loss_fn(out.squeeze(), labels.float())

                # backpropagate error
                loss.backward()
                # clip gradients
                gradient_clipping(model, clip=cf['max_grad_norm'])
                # learn
                optimizer.step()

                av_score.append(loss.to('cpu').detach().numpy())
                print('\r error: {:.10f} '.format(loss.item()))

            loss_dict['train_loss'].append(loss.to('cpu').detach().numpy())

            val_loss = list()
            model.eval()
            for batch in val_dataloader:
                print('--- validation ...')
                with torch.no_grad():
                    sn_input_ids = batch['sn_input_ids'].to(device)
                    sn_attention_mask = batch['sn_attention_mask'].to(device)
                    word_ids_sn = batch['word_ids_sn'].to(device)
                    sn_word_len = batch['sn_word_len'].to(device)

                    sp_input_ids = batch['sp_input_ids'].to(device)
                    sp_attention_mask = batch['sp_attention_mask'].to(device)
                    word_ids_sp = batch['word_ids_sp'].to(device)
                    sp_len = batch['sp_len'].to(device)

                    sp_pos = batch['sp_pos'].to(device)
                    sp_fix_dur = (batch['sp_fix_dur'] / 1000).to(device)

                    # normalize gaze features
                    mask = ~torch.eq(sp_fix_dur, 0)
                    sp_fix_dur = (sp_fix_dur - fix_dur_mean) / fix_dur_std * mask
                    sp_fix_dur = torch.nan_to_num(sp_fix_dur)
                    sn_word_len = (sn_word_len - sn_word_len_mean) / sn_word_len_std
                    sn_word_len = torch.nan_to_num(sn_word_len)

                    out = model(
                        sn_input_ids=sn_input_ids,
                        sn_attention_mask=sn_attention_mask,
                        sp_input_ids=sp_input_ids,
                        sp_pos=sp_pos,
                        word_ids_sn=word_ids_sn,
                        word_ids_sp=word_ids_sp,
                        sp_fix_dur=sp_fix_dur,
                        sn_word_len=sn_word_len,
                        sp_len=sp_len,
                    )
                    if args.loss == 'cross-entropy':
                        loss = loss_fn(out, labels.float())
                    elif args.loss == 'ordinal-hinge':
                        loss = loss_fn(out, labels)
                    elif args.loss == 'mse':
                        loss = loss_fn(out.squeeze(), labels.float())
                    val_loss.append(loss.to('cpu').detach().numpy())












if __name__ == '__main__':
    raise SystemExit(main())
