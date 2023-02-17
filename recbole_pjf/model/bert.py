# @Time   : 2022/3/23
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
pjfbole
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class BERT(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BERT, self).__init__(config, dataset)
        self.USER_SENTS = config['USER_DOC_FIELD']
        self.ITEM_SENTS = config['ITEM_DOC_FIELD']
        self.neg_prefix = config['NEG_PREFIX']
        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']

        # bert part
        self.bert_user = nn.Linear(self.embedding_size, self.hidden_size)
        self.bert_item = nn.Linear(self.embedding_size, self.hidden_size)

        self.predict_layer = nn.Linear(2 * self.hidden_size, 1)

        self.loss = BPRLoss()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, user, item):
        u = self.bert_user(user.float())
        i = self.bert_item(item.float())
        u_i = torch.cat([u, i], dim=1)
        score = self.predict_layer(u_i)
        return score.squeeze()

    def calculate_loss(self, interaction):
        user = interaction[self.USER_SENTS + '_vec']
        item = interaction[self.ITEM_SENTS + '_vec']
        neg_item = interaction[self.neg_prefix + self.ITEM_SENTS + '_vec']
        pos_socre = self.forward(user, item)
        neg_score = self.forward(user, neg_item)

        loss = self.loss(pos_socre, neg_score)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_SENTS + '_vec']
        item = interaction[self.ITEM_SENTS + '_vec']

        score = self.forward(user, item)
        return score
