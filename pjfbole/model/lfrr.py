# @Time   : 2022/3/23
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
pjfbole
"""

import numpy as np
import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class LFRR(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LFRR, self).__init__(config, dataset)
        self.neg_user_id = self.NEG_USER_ID = config['NEG_PREFIX'] + self.USER_ID
        self.embedding_size = config['embedding_size']

        # define layers and loss
        self.user_embedding_1 = nn.Embedding(self.n_users, self.embedding_size)
        self.user_embedding_2 = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding_1 = nn.Embedding(self.n_items, self.embedding_size)
        self.item_embedding_2 = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward_ui(self, user, item):
        u_1 = self.user_embedding_1(user)
        i_1 = self.item_embedding_1(item)

        s_ui = torch.mul(u_1, i_1).sum(dim=1)
        return s_ui

    def forward_iu(self, user, item):
        u_2 = self.user_embedding_2(user)
        i_2 = self.item_embedding_2(item)

        s_iu = torch.mul(u_2, i_2).sum(dim=1)
        return s_iu

    def forward(self, user, item):
        s_ui = self.forward_ui(user, item)
        s_iu = self.forward_iu(user, item)
        score = s_ui + s_iu
        return score

    def calculate_loss(self, interaction):
        pos_user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        neg_user = interaction[self.NEG_USER_ID]

        score_pos = self.forward(pos_user, pos_item)
        score_neg_1 = self.forward(pos_user, neg_item)
        score_neg_2 = self.forward(neg_user, pos_item)

        loss = self.loss(score_pos, score_neg_1) + self.loss(score_pos, score_neg_2)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

