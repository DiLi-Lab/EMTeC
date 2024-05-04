#!/usr/bin/env python3
"""
Train the Eyettention architecture on a classification task of subjective text difficulty.
"""

from analyses.utils.utils_analyses import prepare_eyettention_input, EMTeCDataset, calculate_mean_std
from transformers import BertTokenizerFast
from argparse import ArgumentParser
from analyses.utils.eyettention_model import Eyettention, ClassificationModel
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn



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
    }

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

    emtec_dataset = EMTeCDataset(data)
    emtec_dataloader = DataLoader(emtec_dataset, batch_size=cf['batch_size'], shuffle=True, drop_last=True)

    # model = Eyettention(cf=cf)
    # model.to(device)

    # z-score normalization for gaze features
    fix_dur_mean, fix_dur_std = calculate_mean_std(
        dataloader=emtec_dataloader,
        feat_key='sp_fix_dur',
        padding_value=0,
        scale=1000,
    )
    sn_word_len_mean, sn_word_len_std = calculate_mean_std(
        dataloader=emtec_dataloader,
        feat_key='sn_word_len',
    )

    model = ClassificationModel(
        cf=cf,
        device=device,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cf['lr'])

    for batch_idx, batch in enumerate(emtec_dataloader):

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

        labels = batch['rating_difficulty_one_hot'].to(device)

        # fixation_representation = model(
        #     sn_emd=sn_input_ids,
        #     sn_mask=sn_attention_mask,
        #     sp_emd=sp_input_ids,
        #     sp_pos=sp_pos,
        #     word_ids_sn=word_ids_sn,
        #     word_ids_sp=word_ids_sp,
        #     sp_fix_dur=sp_fix_dur,
        #     sn_word_len=sn_word_len,
        # )


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

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(out, labels.float())

        print(loss)

        loss.backward()
        optimizer.step()

        # lstm = nn.LSTM(
        #     input_size=fixation_representation.size(-1),
        #     hidden_size=cf['hidden_size'],
        #     num_layers=4,
        #     batch_first=True,
        #     dropout=0.2,
        #     bidirectional=True,
        # )
        # # h_n is of shape (num_layers * num_directions, batch, hidden_size)
        # # it contains the final hidden state (concatenated, if bidirectional) for each LSTM layer
        # lstm_out, (h_n, c_n) = lstm(fixation_representation)
        # # access the last hidden state of the last layer for each element in the batch
        # # we first need to reshape h_n to (num_layers, batch, num_directions * hidden_size)
        #
        # # Reshape to (num_layers, num_directions, batch, hidden_size)
        # h_n = h_n.view(cf['num_lstm_layers'], 2, cf['batch_size'], cf['hidden_size'])
        # # Swap dimensions to (num_layers, batch, num_directions, hidden_size)
        # h_n = h_n.permute(0, 2, 1, 3)
        # # Reshape to (num_layers, batch, 2 * hidden_size)
        # h_n = h_n.contiguous().view(cf['num_lstm_layers'], cf['batch_size'], -1)







if __name__ == '__main__':
    raise SystemExit(main())
