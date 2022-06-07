# @Time   : 2022/3/18
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole.model.layers import MLPLayers

from torch.nn.init import xavier_normal_, xavier_uniform_


class BPJFNN(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BPJFNN, self).__init__(config, dataset)
        self.USER_SENTS = config['USER_DOC_FIELD']
        self.ITEM_SENTS = config['ITEM_DOC_FIELD']
        self.neg_prefix = config['NEG_PREFIX']
        self.embedding_size = config['embedding_size']
        self.hd_size = config['hidden_size']
        self.dropout = config['dropout']
        self.num_layers = config['num_layers']
        self.wd_num = len(dataset.wd2id.keys())

        # define layers and loss
        self.emb = nn.Embedding(self.wd_num, self.embedding_size, padding_idx=0)
        self.job_biLSTM = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.geek_biLSTM = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.mlp = MLPLayers(
            layers=[self.hd_size * 3 * 2, self.hd_size, 1],
            dropout=self.dropout,
            activation='tanh'
        )

        self.sigmoid = nn.Sigmoid()
        # self.loss = nn.BCEWithLogitsLoss()
        self.loss = BPRLoss()

    def _single_bpj_layer(self, interaction, token, field):
        longsent = interaction[field]
        vec = self.emb(longsent)
        vec, _ = getattr(self, f'{token}_biLSTM')(vec)
        vec = torch.sum(vec, dim=1) / len(vec)
        return vec

    def forward(self, interaction, geek_field, job_field):
        geek_vec = self._single_bpj_layer(interaction, 'geek', geek_field)
        job_vec = self._single_bpj_layer(interaction, 'job', job_field)

        x = torch.cat([job_vec, geek_vec, job_vec - geek_vec], dim=1)
        x = self.mlp(x).squeeze(1)
        return x

    def calculate_loss(self, interaction):
        output_pos = self.forward(interaction, 'long_' + self.USER_SENTS, 'long_' + self.ITEM_SENTS)
        output_neg = self.forward(interaction, 'long_' + self.USER_SENTS, self.neg_prefix + 'long_' + self.ITEM_SENTS)
        return self.loss(output_pos, output_neg)

    def predict(self, interaction):
        score = self.forward(interaction, 'long_' + self.USER_SENTS, 'long_' + self.ITEM_SENTS)
        return self.sigmoid(score)

