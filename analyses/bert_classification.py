#!/usr/bin/env python3
"""
Train a transformer to predict subjective text difficulty, engaging with the text, etc. from reading measures.
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from analyses.utils.utils_analyses import prepare_bert_input, EMTeCBert, TransformerModel
from analyses.utils.utils_analyses import get_kfold, subset_data_kfold, split_train_val_bert
from analyses.utils.utils_analyses import gradient_clipping
from torch.utils.data import DataLoader
import torch.nn as nn
from argparse import ArgumentParser
from collections import deque
from sklearn.metrics import roc_auc_score
import pickle


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--target',
        type=str,
        default='difficulty',
        choices=['difficulty', 'engaging', 'text-type'],
        help='Whether we try to predict subjective text difficulty or engagement of the text.',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='subject',
        choices=['subject', 'random'],
        help='How to split the data for cross-validation.',
    )
    parser.add_argument(
        '--bsz',
        default=16,
        type=int,
        help='The batch size to use for training and testing.',
    )
    parser.add_argument(
        '--normalized',
        default=False,
        action='store_true',
        help='Whether we use the labels as they are or whether they are z-score normalized.',
    )
    return parser


def main():

    args = get_parser().parse_args()

    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        device = f'cuda:{device_idx}'
    else:
        device = 'cpu'

    #device = 'cpu'

    path_to_rms = 'data/reading_measures_corrected.csv'
    path_to_ratings = 'data/participant_info/participant_results.csv'
    path_to_stimuli = 'data/stimuli.csv'

    exclude_subjects = ['ET_03', 'ET_11', 'ET_39', 'ET_49', 'ET_67', 'ET_83']

    if args.target == 'text-type':
        n_targets = 6
    else:
        if args.normalized:
            n_targets = 1
        else:
            n_targets = 5

    # sanity check
    if args.normalized and args.target == 'text-type':
        raise RuntimeError('use classification for prediction of text type.')

    config = {
        'max_sn_len': 132,
        'lr': 1e-5,
        'batch_size': args.bsz,
        'n_folds': 5,
        'n_epochs': 1000,
        'earlystop_patience': 20,
        'n_targets': n_targets,
        'split': args.split,
        'target': args.target,
        'normalized': args.normalized,
        'max_grad_norm': 10,
    }

    # save paths
    model_save_basepath = '/srv/scratch1/bolliger/EMTeC/analyses/bert_classification'
    model_name = f'{args.target}-{args.split}-{args.bsz}-{args.normalized}'
    model_savepath = os.path.join(model_save_basepath, model_name)
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)

    data = prepare_bert_input(
        path_to_rms=path_to_rms,
        path_to_ratings=path_to_ratings,
        path_to_stimuli=path_to_stimuli,
        exclude_subjects=exclude_subjects,
        max_sn_len=config['max_sn_len'],
    )

    # which label to use
    label_key = ''
    label_key_onehot = ''
    task = ''
    if args.target == 'text-type':
        label_key = 'text_types_int'
        label_key_onehot = 'text_types_onehot'
        task = 'classification'
    elif args.target == 'difficulty':
        if args.normalized:
            label_key = 'difficulty_zscore_labels'
            task = 'regression'
        else:
            label_key = 'difficulty_labels'
            label_key_onehot = 'difficulty_onehot_labels'
            task = 'classification'
    elif args.target == 'engaging':
        if args.normalized:
            label_key = 'engaging_zscore_labels'
            task = 'regression'
        else:
            label_key = 'engaging_labels'
            label_key_onehot = 'engaging_onehot_labels'
            task = 'classification'


    # iterate through the folds
    for fold_idx, (train_idx, test_idx) in enumerate(
        get_kfold(
            inputs=data['features'],
            labels=data[label_key],
            split=config['split'],
            n_splits=config['n_folds'],
            group=data['subject_ids'],
        )
    ):

        # TODO remove break
        if fold_idx == 1:
            break

        # dict to hold the training and test stats
        stats_dict = {
            'train_loss': list(),
            'val_loss': list(),
            'test_loss': list(),
            #'test_AUC': list(),
            #'true_test_labels_regression': list(),
            #'pred_test_labels_regression': list(),
            #'true_test_labels_classification': list(),
            #'pred_test_labels_classification': list(),
        }

        # split the data into train and test fold
        train_data, test_data = subset_data_kfold(
            data=data,
            train_idx=train_idx,
            test_idx=test_idx,
        )

        # split the train data into train and validation data
        train_data, val_data = split_train_val_bert(
            train_data=train_data,
        )

        # wrap in dataset class
        train_dataset = EMTeCBert(data=train_data)
        test_dataset = EMTeCBert(data=test_data)
        val_dataset = EMTeCBert(data=val_data)

        # wrap in dataloader iterator
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=False)
        val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)

        # model, optimizer and loss function
        model = TransformerModel(num_classes=config['n_targets'])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        if args.target == 'text-type':
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        elif args.target == 'difficulty':
            if args.normalized:
                loss_fn = torch.nn.MSELoss()
            else:
                loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        elif args.target == 'engaging':
            if args.normalized:
                loss_fn = torch.nn.MSELoss()
            else:
                loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        # variables for early stopping
        old_score = 1e10
        av_score = deque(maxlen=100)
        save_ep_counter = 0

        for epoch in range(config['n_epochs']):
            model.train()

            print(model_savepath)

            for batch_idx, batch in enumerate(train_dataloader):


                print(f'--- fold {fold_idx} training epoch {epoch} batch {batch_idx}', end='\t')

                features = batch['features'].to(device)
                mask = batch['mask'].to(device)
                labels = batch[label_key].to(device)

                optimizer.zero_grad()

                out = model(
                    features=features,
                    attention_mask=mask,
                )


                if task == 'regression':
                    loss = loss_fn(out.squeeze(), labels)
                else:
                    loss = loss_fn(out, labels)


                loss.backward()
                gradient_clipping(model, clip=config['max_grad_norm'])
                optimizer.step()

                av_score.append(loss.to('cpu').detach().numpy())
                print(f'error: {loss.to("cpu").detach().numpy()}')

                stats_dict['train_loss'].append(loss.to('cpu').detach().numpy().item())

            val_loss = list()
            model.eval()
            for batch_idx, batch in enumerate(val_dataloader):

                print(f'--- fold {fold_idx} validation epoch {epoch} batch {batch_idx}')
                with torch.no_grad():
                    features = batch['features'].to(device)
                    mask = batch['mask'].to(device)
                    labels = batch[label_key].to(device)

                    out = model(
                        features=features,
                        attention_mask=mask,
                    )
                    if task == 'regression':
                        loss = loss_fn(out.squeeze(), labels)
                    else:
                        loss = loss_fn(out, labels)
                    val_loss.append(loss.to('cpu').detach().numpy())
                    stats_dict['val_loss'].append(loss.to('cpu').detach().numpy().item())

            # check if early stopping should be applied
            if np.mean(val_loss) < old_score:
                # save model if val loss is smallest
                torch.save(model.state_dict(), os.path.join(model_savepath, f'model-fold{fold_idx}.pth'))
                old_score = np.mean(val_loss)
                print(f'\t\tsaved model state dict')
                save_ep_counter = epoch
            else:
                # early stopping
                if epoch - save_ep_counter >= config['earlystop_patience']:
                    stats_dict['early_stopping'] = True
                    break

        # evaluation
        # load model at best epoch
        print(model_savepath)
        model.load_state_dict(torch.load(os.path.join(model_savepath, f'model-fold{fold_idx}.pth'), map_location='cpu'))
        model.to(device)
        model.eval()
        test_outputs, test_labels, test_labels_onehot = list(), list(), list()
        models, decoding_strategies, item_ids = list(), list(), list()
        flesch_scores, flesch_kincaid_scores, gunning_fog_scores = list(), list(), list()
        coleman_liau_scores, dale_chall_scores, ari_scores = list(), list(), list()
        linsear_write_scores, spache_scores = list(), list()

        for batch_idx, batch in enumerate(test_dataloader):

            with torch.no_grad():
                print(f'--- fold {fold_idx} testing batch {batch_idx}')

                features = batch['features'].to(device)
                mask = batch['mask'].to(device)
                labels = batch[label_key].to(device)

                out = model(
                    features=features,
                    attention_mask=mask,
                )

                if task == 'regression':
                    loss = loss_fn(out.squeeze(), labels)
                else:
                    loss = loss_fn(out, labels)
                stats_dict['test_loss'].append(loss.to('cpu').detach().numpy().item())

                if task == 'classification':
                    test_outputs.append(out.to('cpu').detach().numpy())
                    test_labels.append(labels.to('cpu').detach().numpy())
                    test_labels_onehot.append(batch[label_key_onehot].to('cpu').detach().numpy())

                if task == 'regression':
                    for l in labels:
                        test_labels.append(l.item())
                    for o in out.squeeze(1):
                        test_outputs.append(o.item())

                # add all meta information to list
                model_name = batch['model']
                decoding_strategy = batch['decoding_strategy']
                item_id = batch['item_id']
                flesch = batch['flesch'].tolist()
                flesch_kincaid = batch['flesch_kincaid'].tolist()
                gunning_fog = batch['gunning_fog'].tolist()
                coleman_liau = batch['coleman_liau'].tolist()
                dale_chall = batch['dale_chall'].tolist()
                ari = batch['ari'].tolist()
                linsear_write = batch['linsear_write'].tolist()
                spache = batch['spache'].tolist()

                models.extend(model_name)
                decoding_strategies.extend(decoding_strategy)
                item_ids.extend(item_id)
                flesch_scores.extend(flesch)
                flesch_kincaid_scores.extend(flesch_kincaid)
                gunning_fog_scores.extend(gunning_fog)
                coleman_liau_scores.extend(coleman_liau)
                dale_chall_scores.extend(dale_chall)
                ari_scores.extend(ari)
                linsear_write_scores.extend(linsear_write)
                spache_scores.extend(spache)


        if task == 'classification':

            # flatten the array of test outputs from [n batches, batch size, n classes] into [n samples, n classes]
            all_test_outputs_flattened = np.concatenate(test_outputs, axis=0)
            # convert the logits into probabilities
            probabilities = nn.functional.softmax(torch.tensor(all_test_outputs_flattened), dim=1)
            # flatten the true one-hot encoded test labels
            labels_onehot_flattened = np.concatenate(test_labels_onehot, axis=0)
            # compute the AUC score
            auc_score = roc_auc_score(y_true=labels_onehot_flattened, y_score=probabilities)
            #all_test_outputs = torch.cat([t.view(-1, n_targets) for t in test_outputs], dim=0)
            #all_test_labels = torch.cat([t.view(-1, n_targets) for t in test_labels], dim=0)
            #probabilities = nn.functional.softmax(all_test_outputs, dim=1).cpu()
            #auc_score = roc_auc_score(test_labels, probabilities[:, test_labels])

            # get the actual predicted classes/predicted indices of the model
            pred_classes = torch.argmax(probabilities, dim=1)
            # flatten the true test labels (same as torch argmaxing the one-hot encoded labels)
            test_labels_flattened = torch.tensor(np.concatenate(test_labels, axis=0))

            # add AUC score, predicted labels, true labels, etc. to stats dict
            stats_dict['test_AUC'] = auc_score
            stats_dict['true_labels_classification'] = test_labels_flattened.tolist()
            stats_dict['pred_labels_classification'] = pred_classes.tolist()
            stats_dict['true_labels_onehot_classification'] = labels_onehot_flattened
            stats_dict['pred_logits_classification'] = all_test_outputs_flattened
            stats_dict['pred_probabilities_classification'] = probabilities

        if task == 'regression':
            stats_dict['true_labels_regression'] = test_outputs
            stats_dict['pred_labels_regression'] = test_labels


            # stats_dict['test_AUC'].append(auc_score)
            # for test_label in test_labels:
            #     stats_dict['true_test_labels_classification'].append(test_label)
            # predicted_classes = torch.argmax(probabilities, dim=1)
            # for pred_class in predicted_classes:
            #     stats_dict['pred_test_labels_classification'].append(pred_class)

        # add all metadata to stats dict
        stats_dict['model'] = models
        stats_dict['decoding_strategy'] = decoding_strategies
        stats_dict['item_id'] = item_ids
        stats_dict['flesch'] = flesch_scores
        stats_dict['flesch_kincaid'] = flesch_kincaid_scores
        stats_dict['gunning_fog'] = gunning_fog_scores
        stats_dict['coleman_liau'] = coleman_liau_scores
        stats_dict['dale_chall'] = dale_chall_scores
        stats_dict['ari'] = ari_scores
        stats_dict['linsear_write'] = linsear_write_scores
        stats_dict['spache'] = spache_scores

        # save results
        with open(os.path.join(model_savepath, f'model-results-fold{fold_idx}.pickle'), 'wb') as handle:
            pickle.dump(stats_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # save config
    with open(os.path.join(model_savepath, 'config.pickle'), 'wb') as handle:
        pickle.dump(config, handle)




if __name__ == '__main__':
    raise SystemExit(main())
