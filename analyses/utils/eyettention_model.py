#!/usr/bin/env python3
"""
The Eyettention model, adapted to return the concatenation of the last hidden state and the cross-attention output.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax
from transformers import BertModel, BertConfig
from typing import Optional, Dict, Any


class Eyettention(nn.Module):
    def __init__(self, cf):
        super(Eyettention, self).__init__()
        self.cf = cf
        self.window_width = 1  # D
        self.atten_type = cf["atten_type"]
        self.hidden_size = cf["hidden_size"]
        self.eyettention_output = cf['eyettention_output']

        # Word-Sequence Encoder
        encoder_config = BertConfig.from_pretrained(self.cf["model_pretrained"])
        encoder_config.output_hidden_states = True
        # initiate Bert with pre-trained weights
        print("keeping Bert with pre-trained weights")
        self.encoder = BertModel.from_pretrained(self.cf["model_pretrained"], config=encoder_config)
        self.encoder.eval()
        # freeze the parameters in Bert model
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.embedding_dropout = nn.Dropout(0.4)
        self.encoder_lstm = nn.LSTM(input_size=768,  # BERT embedding size
                                    hidden_size=int(self.hidden_size / 2),
                                    num_layers=8,
                                    batch_first=True,
                                    bidirectional=True,
                                    dropout=0.2)

        # Fixation-Sequence Encoder
        self.position_embeddings = nn.Embedding(encoder_config.max_position_embeddings, encoder_config.hidden_size)
        self.LayerNorm = nn.LayerNorm(encoder_config.hidden_size, eps=encoder_config.layer_norm_eps)

        # The scanpath is generated in an autoregressive manner, the output of the previous timestep is fed to the input of the next time step.
        # So we use decoder cells and loop over all timesteps.
        # initialize eight decoder cells
        #self.decoder_cell1 = nn.LSTMCell(768 + 2, self.hidden_size)  # first layer input size = #BERT embedding size + two fixation attributes:landing position and fixiation duration
        # only + 1 because we don't have sp_landing_pos but only sp_fix_dur
        self.decoder_cell1 = nn.LSTMCell(768 + 1, self.hidden_size)
        self.decoder_cell2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell4 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell5 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell6 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell7 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell8 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.dropout_LSTM = nn.Dropout(0.2)

        # Cross-Attention
        self.attn = nn.Linear(self.hidden_size, self.hidden_size + 1)  # +1 acount for the word length feature

        # Decoder
        # initialize five dense layers
        self.dropout_dense = nn.Dropout(0.2)
        self.decoder_dense = nn.Sequential(
            self.dropout_dense,
            nn.Linear(self.hidden_size * 2 + 1, 512),
            nn.ReLU(),
            self.dropout_dense,
            nn.Linear(512, 256),
            nn.ReLU(),
            self.dropout_dense,
            nn.Linear(256, 256),
            nn.ReLU(),
            self.dropout_dense,
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.cf["max_sn_len"] * 2 - 3),  # number of output classes
        )

        # for scanpath generation
        self.softmax = nn.Softmax(dim=1)

    def pool_subword_to_word(self, subword_emb, word_ids_sn, target, pool_method='sum'):
        # batching computing
        # Pool bert token (subword) to word level
        if target == 'sn':
            max_len = self.cf["max_sn_len"]  # CLS and SEP included
        elif target == 'sp':
            #max_len = self.cf["max_sp_len"] - 1  # do not account the 'SEP' token
            max_len = self.cf['max_sp_len']

        merged_word_emb = torch.empty(subword_emb.shape[0], 0, 768).to(subword_emb.device)
        for word_idx in range(max_len):
            word_mask = (word_ids_sn == word_idx).unsqueeze(2).repeat(1, 1, 768)
            # pooling method -> sum
            if pool_method == 'sum':
                pooled_word_emb = torch.sum(subword_emb * word_mask, 1).unsqueeze(1)  # [batch, 1, 768]
            elif pool_method == 'mean':
                pooled_word_emb = torch.mean(subword_emb * word_mask, 1).unsqueeze(1)  # [batch, 1, 768]
            merged_word_emb = torch.cat([merged_word_emb, pooled_word_emb], dim=1)

        mask_word = torch.sum(merged_word_emb, 2).bool()
        return merged_word_emb, mask_word

    def encode(self, sn_emd, sn_mask, word_ids_sn, sn_word_len):
        # Word-Sequence Encoder
        outputs = self.encoder(input_ids=sn_emd, attention_mask=sn_mask)
        hidden_rep_orig, pooled_rep = outputs[0], outputs[1]
        if word_ids_sn != None:
            # Pool bert subword to word level for english corpus
            merged_word_emb, sn_mask_word = self.pool_subword_to_word(hidden_rep_orig,
                                                                      word_ids_sn,
                                                                      target='sn',
                                                                      pool_method='sum')
        else:  # no pooling for Chinese corpus
            merged_word_emb, sn_mask_word = hidden_rep_orig, None

        hidden_rep = self.embedding_dropout(merged_word_emb)
        x, (hn, hc) = self.encoder_lstm(hidden_rep, None)

        # concatenate with the word length feature
        x = torch.cat((x, sn_word_len[:, :, None]), dim=2)
        return x, sn_mask_word

    def cross_attention(self, ht, hs, sn_mask, cur_word_index):
        # General Attention:
        # score(ht,hs) = (ht^T)(Wa)hs
        # hs is the output from word-Sequence Encoder
        # ht is the previous hidden state from Fixation-Sequence Encoder
        # self.attn(o): [batch, step, units]
        attn_prod = torch.matmul(self.attn(ht.unsqueeze(1)), hs.permute(0, 2, 1))  # [batch, 1, step]
        if self.atten_type == 'global':  # global attention
            attn_prod += (~sn_mask).unsqueeze(1) * -1e9
            att_weight = softmax(attn_prod, dim=2)  # [batch, 1, step]

        else:  # local attention
            # current fixated word index
            aligned_position = cur_word_index
            # Get window borders
            left = torch.where(aligned_position - self.window_width >= 0, aligned_position - self.window_width, 0)
            right = torch.where(aligned_position + self.window_width <= self.cf["max_sn_len"] - 1,
                                aligned_position + self.window_width, self.cf["max_sn_len"] - 1)

            # exclude padding tokens
            # only consider words in the window
            sen_seq = torch.arange(self.cf["max_sn_len"])[None, :].expand(sn_mask.shape[0], self.cf["max_sn_len"]).to(
                sn_mask.device)
            outside_win_mask = (sen_seq < left.unsqueeze(1)) + (sen_seq > right.unsqueeze(1))

            attn_prod += (~sn_mask + outside_win_mask).unsqueeze(1) * -1e9
            att_weight = softmax(attn_prod, dim=2)  # [batch, 1, step]

            if self.atten_type == 'local-g':  # local attention with Gaussian Kernel
                gauss = lambda s: torch.exp(-torch.square(s - aligned_position.unsqueeze(1)) / (
                            2 * torch.square(torch.tensor(self.window_width / 2))))
                gauss_factor = gauss(sen_seq)
                att_weight = att_weight * gauss_factor.unsqueeze(1)

        return att_weight

    def decode(self, sp_emd, sn_mask, sp_pos, enc_out, sp_fix_dur, sp_landing_pos, word_ids_sp):
        # Fixation-Sequence Encoder + Decoder
        # Initialize hidden state and cell state with zeros,
        hn = torch.zeros(8, sp_emd.shape[0], self.hidden_size).to(sp_emd.device)
        hc = torch.zeros(8, sp_emd.shape[0], self.hidden_size).to(sp_emd.device)
        hx, cx = hn[0, :, :], hc[0, :, :]
        hx2, cx2 = hn[1, :, :], hc[1, :, :]
        hx3, cx3 = hn[2, :, :], hc[2, :, :]
        hx4, cx4 = hn[3, :, :], hc[3, :, :]
        hx5, cx5 = hn[4, :, :], hc[4, :, :]
        hx6, cx6 = hn[5, :, :], hc[5, :, :]
        hx7, cx7 = hn[6, :, :], hc[6, :, :]
        hx8, cx8 = hn[7, :, :], hc[7, :, :]

        dec_emb_in = self.encoder.embeddings.word_embeddings(sp_emd[:, :-1])
        if word_ids_sp is not None:
            # Pool bert subword to word level for English corpus
            sp_merged_word_emd, sp_mask_word = self.pool_subword_to_word(dec_emb_in,
                                                                         word_ids_sp[:, :-1],
                                                                         target='sp',
                                                                         pool_method='sum')
        else:  # no pooling for Chinese corpus
            sp_merged_word_emd, sp_mask_word = dec_emb_in, None

        # add positional embeddings
        #position_embeddings = self.position_embeddings(sp_pos[:, :-1])
        position_embeddings = self.position_embeddings(sp_pos)
        dec_emb_in = sp_merged_word_emd + position_embeddings
        dec_emb_in = self.LayerNorm(dec_emb_in)
        dec_emb_in = dec_emb_in.permute(1, 0, 2)  # [step, n, emb_dim]
        dec_emb_in = self.embedding_dropout(dec_emb_in)

        # concatenate two additional gaze features
        if sp_landing_pos is not None:
            dec_emb_in = torch.cat((dec_emb_in, sp_landing_pos.permute(1, 0)[:-1, :, None]), dim=2)

        if sp_fix_dur is not None:
            #dec_emb_in = torch.cat((dec_emb_in, sp_fix_dur.permute(1, 0)[:-1, :, None]), dim=2)
            dec_emb_in = torch.cat((dec_emb_in, sp_fix_dur.permute(1, 0)[:, :, None]), dim=2)

        # Predict output for each time step in turn
        output = []
        # save attention scores for visualization
        atten_weights_batch = torch.empty(sp_emd.shape[0], 0, self.cf["max_sn_len"]).to(sp_emd.device)

        scanpath_representation = torch.tensor

        for i in range(dec_emb_in.shape[0]):
            hx, cx = self.decoder_cell1(dec_emb_in[i], (hx, cx))  # [batch, units]
            hx2, cx2 = self.decoder_cell2(self.dropout_LSTM(hx), (hx2, cx2))
            hx3, cx3 = self.decoder_cell3(self.dropout_LSTM(hx2), (hx3, cx3))
            hx4, cx4 = self.decoder_cell4(self.dropout_LSTM(hx3), (hx4, cx4))
            hx5, cx5 = self.decoder_cell5(self.dropout_LSTM(hx4), (hx5, cx5))
            hx6, cx6 = self.decoder_cell6(self.dropout_LSTM(hx5), (hx6, cx6))
            hx7, cx7 = self.decoder_cell7(self.dropout_LSTM(hx6), (hx7, cx7))
            hx8, cx8 = self.decoder_cell8(self.dropout_LSTM(hx7), (hx8, cx8))

            att_weight = self.cross_attention(ht=hx8,
                                              hs=enc_out,
                                              sn_mask=sn_mask,
                                              cur_word_index=sp_pos[:, i])
            atten_weights_batch = torch.cat([atten_weights_batch, att_weight], dim=1)

            context = torch.matmul(att_weight, enc_out)  # [batch, 1, units]

            # Decoder
            hc = torch.cat([context.squeeze(1), hx8], dim=1)  # [batch, units *2]
            result = self.decoder_dense(hc)  # [batch, dec_o_dim]
            output.append(result)

            if i == 0:
                if self.eyettention_output == 'last_hidden':
                    scanpath_representation = hx8.unsqueeze(1)
                elif self.eyettention_output == 'context':
                    scanpath_representation = context
                elif self.eyettention_output == 'both_concat':
                    scanpath_representation = hc.unsqueeze(1)
                else:
                    raise NotImplementedError(f"Classification input type {self.classification_input} not implemented.")
            else:
                if self.eyettention_output == 'last_hidden':
                    scanpath_representation = torch.cat([scanpath_representation, hx8.unsqueeze(1)], dim=1)
                elif self.eyettention_output == 'context':
                    scanpath_representation = torch.cat([scanpath_representation, context], dim=1)
                elif self.eyettention_output == 'both_concat':
                    scanpath_representation = torch.cat([scanpath_representation, hc.unsqueeze(1)], dim=1)
                else:
                    raise NotImplementedError(f"Classification input type {self.classification_input} not implemented.")

        output = torch.stack(output, dim=0)  # [step, batch, dec_o_dim]
        # output = F.softmax(output, dim=2) # cross entropy in pytorch includes softmax
        #return output.permute(1, 0, 2), atten_weights_batch  # [batch, step, dec_o_dim]
        return scanpath_representation

    def forward(
            self,
            sn_emd,
            sn_mask,
            sp_emd,
            sp_pos,
            word_ids_sn,
            word_ids_sp,
            sp_fix_dur,
            sn_word_len,
            sp_landing_pos: Optional[torch.Tensor] = None,
    ):
        x, sn_mask_word = self.encode(sn_emd, sn_mask, word_ids_sn, sn_word_len)  # [batch, step, units], [batch, units]

        if sn_mask_word is None:  # for Chinese dataset without token pooling
            sn_mask = torch.Tensor.bool(sn_mask)
            # pred, atten_weights = self.decode(sp_emd,
            #                                   sn_mask,
            #                                   sp_pos,
            #                                   x,
            #                                   sp_fix_dur,
            #                                   sp_landing_pos,
            #                                   word_ids_sp)  # [batch, step, dec_o_dim]
            scanpath_representation = self.decode(
                sp_emd,
                sn_mask,
                sp_pos,
                x,
                sp_fix_dur,
                sp_landing_pos,
                word_ids_sp,
            )

        else:  # for English dataset with token pooling
            # pred, atten_weights = self.decode(sp_emd,
            #                                   sn_mask_word,
            #                                   sp_pos,
            #                                   x,
            #                                   sp_fix_dur,
            #                                   sp_landing_pos,
            #                                   word_ids_sp)  # [batch, step, dec_o_dim]
            scanpath_representation = self.decode(
                sp_emd,
                sn_mask_word,
                sp_pos,
                x,
                sp_fix_dur,
                sp_landing_pos,
                word_ids_sp,
            )

        #return pred, atten_weights
        return scanpath_representation

    def scanpath_generation(self, sn_emd,
                            sn_mask,
                            word_ids_sn,
                            sn_word_len,
                            le,
                            max_pred_len=60):
        # compute the scan path generated from the model when the first CLS taken is given
        enc_out, sn_mask_word = self.encode(sn_emd, sn_mask, word_ids_sn, sn_word_len)
        if sn_mask_word is None:
            sn_mask = torch.Tensor.bool(sn_mask)
        else:
            sn_mask = sn_mask_word
        sn_len = torch.sum(sn_mask, axis=1) - 2

        # decode
        # Initialize hidden state and cell state with zeros,
        hn = torch.zeros(8, sn_emd.shape[0], self.hidden_size).to(sn_emd.device)
        hc = torch.zeros(8, sn_emd.shape[0], self.hidden_size).to(sn_emd.device)
        hx, cx = hn[0, :, :], hc[0, :, :]
        hx2, cx2 = hn[1, :, :], hc[1, :, :]
        hx3, cx3 = hn[2, :, :], hc[2, :, :]
        hx4, cx4 = hn[3, :, :], hc[3, :, :]
        hx5, cx5 = hn[4, :, :], hc[4, :, :]
        hx6, cx6 = hn[5, :, :], hc[5, :, :]
        hx7, cx7 = hn[6, :, :], hc[6, :, :]
        hx8, cx8 = hn[7, :, :], hc[7, :, :]

        # use CLS token (101) as start token
        dec_in_start = (torch.ones(sn_mask.shape[0]) * 101).long().to(sn_mask.device)
        dec_emb_in = self.encoder.embeddings.word_embeddings(dec_in_start)  # [batch, emb_dim]
        # dec_in_start = sp_emd[:, 0]
        # dec_emb_in = self.encoder.embeddings.word_embeddings(dec_in_start) # [batch, emb_dim]

        # add positional embeddings
        start_pos = torch.zeros(sn_mask.shape[0]).to(sn_mask.device)
        position_embeddings = self.position_embeddings(start_pos.long())
        dec_emb_in = dec_emb_in + position_embeddings
        dec_emb_in = self.LayerNorm(dec_emb_in)

        # concatenate two additional gaze features, which are set to zeros for CLS token
        dec_in = torch.cat((dec_emb_in, torch.zeros(dec_emb_in.shape[0], 2).to(sn_emd.device)), dim=1)

        # generate fixation one by one in an autoregressive way
        output = []
        density_prediction = []
        pred_counter = 0
        # output.append(sp_pos[:, pred_counter])
        output.append(start_pos.long())
        for p in range(max_pred_len - 1):
            hx, cx = self.decoder_cell1(dec_in, (hx, cx))  # [batch, units]
            hx2, cx2 = self.decoder_cell2(self.dropout_LSTM(hx), (hx2, cx2))
            hx3, cx3 = self.decoder_cell3(self.dropout_LSTM(hx2), (hx3, cx3))
            hx4, cx4 = self.decoder_cell4(self.dropout_LSTM(hx3), (hx4, cx4))
            hx5, cx5 = self.decoder_cell5(self.dropout_LSTM(hx4), (hx5, cx5))
            hx6, cx6 = self.decoder_cell6(self.dropout_LSTM(hx5), (hx6, cx6))
            hx7, cx7 = self.decoder_cell7(self.dropout_LSTM(hx6), (hx7, cx7))
            hx8, cx8 = self.decoder_cell8(self.dropout_LSTM(hx7), (hx8, cx8))

            att_weight = self.cross_attention(ht=hx8,
                                              hs=enc_out,
                                              sn_mask=sn_mask,
                                              cur_word_index=output[-1])

            context = torch.matmul(att_weight, enc_out)  # [batch, 1, units]
            hc = torch.cat([context.squeeze(1), hx8], dim=1)  # [batch, units *2]

            result = self.decoder_dense(hc)  # [batch, dec_o_dim]
            result = self.softmax(result)  # [batch, dec_o_dim]
            density_prediction.append(result)

            # we can either take argmax or sampling from the output distribution,
            # we do sampling in the paper
            # pred_indx = result.argmax(dim=1)
            # sampling next fixation location according to the distribution
            pred_indx = torch.multinomial(result, 1)
            pred_class = [le.classes_[pred_indx[i]] for i in torch.arange(result.shape[0])]
            pred_class = torch.from_numpy(np.array(pred_class)).to(sn_emd.device)
            # predict fixation word index = last fixation word index + predicted saccade range
            pred_pos = output[-1] + pred_class

            # larger than sentence max length -- set to sentence length+1, i.e. token <'SEP'>
            # prepare the input to the next timstep
            input_ids = []
            for i in range(pred_pos.shape[0]):
                if pred_pos[i] > sn_len[i]:
                    pred_pos[i] = sn_len[i] + 1
                elif pred_pos[i] < 1:
                    pred_pos[i] = 1

                if word_ids_sn is not None:
                    input_ids.append(sn_emd[i, word_ids_sn[i, :] == pred_pos[i]])
                else:
                    input_ids.append(sn_emd[i, pred_pos[i]])
            output.append(pred_pos)

            # prepare next timestamp input token
            pred_counter += 1
            if word_ids_sn is not None:
                # merge tokens
                dec_emb_in = torch.empty(0, 768).to(sn_emd.device)
                for id in input_ids:
                    dec_emb_in = torch.cat(
                        [dec_emb_in, torch.sum(self.encoder.embeddings.word_embeddings(id), axis=0)[None, :]], dim=0)

            else:
                input_ids = torch.stack(input_ids)
                dec_emb_in = self.encoder.embeddings.word_embeddings(input_ids)  # [batch, emb_dim]
            # add positional embeddings
            position_embeddings = self.position_embeddings(output[-1])
            dec_emb_in = dec_emb_in + position_embeddings
            dec_emb_in = self.LayerNorm(dec_emb_in)
            # concatenate two additional gaze features
            dec_in = torch.cat((dec_emb_in, torch.zeros(dec_emb_in.shape[0], 2).to(sn_emd.device)), dim=1)

        output = torch.stack(output, dim=0)  # [step, batch]
        return output.permute(1, 0), density_prediction  # [batch, step]


class ClassificationModel(nn.Module):

    def __init__(
            self,
            cf: Dict[str, Any],
            device: str,
    ):
        super(ClassificationModel, self).__init__()
        self.cf = cf

        self.eyettention = Eyettention(cf=cf)
        self.eyettention.to(device)

        if self.cf['eyettention_output'] == 'both_concat':
            self.classification_input_size = self.cf['hidden_size'] * 2 + 1
        elif self.cf['eyettention_output'] == 'context':
            self.classification_input_size = self.cf['hidden_size'] + 1
        elif self.cf['eyettention_output'] == 'last_hidden':
            self.classification_input_size = self.cf['hidden_size']

        self.bilstm = nn.LSTM(
            input_size=self.classification_input_size,
            hidden_size=cf['hidden_size'],
            num_layers=cf['num_lstm_layers'],
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )

        if self.cf['classification_input'] == 'last':
            self.dense_input = self.classification_input_size
        elif self.cf['classification_input'] == 'all':
            self.dense_input = 2 * cf['hidden_size']

        self.dropout_dense = nn.Dropout(0.2)
        self.dense = nn.Sequential(
            nn.Linear(self.dense_input, 512),
            nn.ReLU(),
            self.dropout_dense,
            nn.Linear(512, 256),
            nn.ReLU(),
            self.dropout_dense,
            nn.Linear(256, 256),
            nn.ReLU(),
            self.dropout_dense,
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, cf['n_targets']),  # number of output classes = 5
        )

    def forward(
            self,
            sn_input_ids,
            sn_attention_mask,
            sp_input_ids,
            sp_pos,
            word_ids_sn,
            word_ids_sp,
            sp_fix_dur,
            sn_word_len,
            sp_len,
    ):

        # shape [batch, max_sp_len, hidden_size]
        out_eyettention = self.eyettention(
            sn_emd=sn_input_ids,
            sn_mask=sn_attention_mask,
            sp_emd=sp_input_ids,
            sp_pos=sp_pos,
            word_ids_sn=word_ids_sn,
            word_ids_sp=word_ids_sp,
            sp_fix_dur=sp_fix_dur,
            sn_word_len=sn_word_len,
        )

        # if we consider all eyettention output states for classification
        if self.cf['classification_input'] == 'all':
            # pipe the output of the eyettention model through the bilstm
            out_lstm, (h_n, c_n) = self.bilstm(out_eyettention)

            # h_n is of shape (num_layers * num_directions, batch, hidden_size)
            # it contains the final hidden state (concatenated, if bidirectional) for each LSTM layer
            # access the last hidden state of the last layer for each element in the batch
            # we first need to reshape h_n to (num_layers, batch, num_directions * hidden_size)

            # Reshape to (num_layers, num_directions, batch, hidden_size)
            h_n = h_n.view(self.cf['num_lstm_layers'], 2, self.cf['batch_size'], self.cf['hidden_size'])
            # Swap dimensions to (num_layers, batch, num_directions, hidden_size)
            h_n = h_n.permute(0, 2, 1, 3)
            # Reshape to (num_layers, batch, 2 * hidden_size)
            dense_input = h_n.contiguous().view(self.cf['num_lstm_layers'], self.cf['batch_size'], -1)
            # take only the concatenated hidden state of the last layer
            dense_input = dense_input[-1, :, :]

        # if we do the classification only based on the last state of the eyettention output
        elif self.cf['classification_input'] == 'last':
            # we need to access the last state of the eyettention output
            # because the outputs are padded to max_sp_len, we need to access the last non-padded state (of last fix.)
            # the resulting tensor has shape [batch, hidden_size]
            # subtract -1 from sp_len because if scanpath has length 100, we index that token at position 99
            dense_input = out_eyettention[torch.arange(out_eyettention.size(0)), sp_len - 1, :]

        # pipe through dense layers
        out = self.dense(dense_input)
        return out




