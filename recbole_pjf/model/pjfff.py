# @Time   : 2022/5/8
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
pjfbole
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class PJFFF(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(PJFFF, self).__init__(config, dataset)
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.HIS_ITEMS = dataset.his_items_field
        self.HIS_ITEMS_LABEL = dataset.his_items_label_field
        self.HIS_USERS = dataset.his_users_field
        self.HIS_USERS_LABEL = dataset.his_users_label_field
        self.NEG_HIS_USERS = config['NEG_PREFIX'] + dataset.his_users_field
        self.NEG_HIS_USERS_LABEL = config['NEG_PREFIX'] + dataset.his_users_label_field
        self.embedding_size = config['embedding_size']
        self.bert_lr = nn.Linear(config['BERT_embedding_size'], config['BERT_output_size'])

        # layers
        self.geek_emb = nn.Embedding(self.n_users, self.embedding_size)
        self.job_emb = nn.Embedding(self.n_items, self.embedding_size)

        self.bert_user = dataset.bert_user.to(config['device'])
        self.bert_item = dataset.bert_item.to(config['device'])

        self.job_biLSTM = nn.LSTM(
            input_size=self.embedding_size * 2 + config['BERT_output_size'] * 2 + 2,
            hidden_size=self.embedding_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.geek_biLSTM = nn.LSTM(
            input_size=self.embedding_size * 2 + config['BERT_output_size'] * 2 + 2,
            hidden_size=self.embedding_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.job_layer = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.geek_layer = nn.Linear(self.embedding_size * 2, self.embedding_size)

        self.loss = BPRLoss()
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def _get_fg_E_user(self, user_id):
        f_e = self.bert_lr(self.bert_user[user_id])
        f_e = torch.cat([f_e, self.geek_emb(user_id)], dim=1)
        return f_e

    def _get_fg_E_item(self, item_id):
        g_e = self.bert_lr(self.bert_item[item_id])
        g_e = torch.cat([g_e, self.job_emb(item_id)], dim=1)
        return g_e

    def _forward_E(self, user_id, item_id):
        f_e = self._get_fg_E_user(user_id)
        g_e = self._get_fg_E_item(item_id)
        score_E = torch.mul(f_e, g_e).sum(dim=1)
        return score_E

    def _get_fg_I_user(self, user_id, his_items, his_label):
        f_e = self._get_fg_E_user(user_id)
        his_g_e = self.bert_lr(self.bert_item[his_items])  # [2048, 100, 32]
        his_g_e = torch.cat([his_g_e, self.job_emb(his_items), his_label.reshape([his_g_e.shape[0], his_g_e.shape[1], -1])], dim=2)
        f_i = torch.cat((his_g_e, f_e.unsqueeze(1).repeat(1, his_g_e.shape[1], 1)), dim=2)
        # _, (f_i, _) = self.geek_biLSTM(f_i)
        # f_i = f_i.reshape([f_i.shape[1], -1])
        f_i, _ = self.geek_biLSTM(f_i)
        f_i = torch.sum(f_i, dim=1)
        f_i = self.geek_layer(f_i)
        return f_i

    def _get_fg_I_item(self, item_id, his_users, his_label):
        g_e = self._get_fg_E_item(item_id)
        his_f_e = self.bert_lr(self.bert_user[his_users])  # [2048, 100, 32]
        his_f_e = torch.cat([his_f_e, self.geek_emb(his_users), his_label.reshape([his_f_e.shape[0], his_f_e.shape[1], -1])], dim=2)
        g_i = torch.cat((his_f_e, g_e.unsqueeze(1).repeat(1, his_f_e.shape[1], 1)), dim=2)  # [2048, 100, 64]
        # _, (g_i, _) = self.job_biLSTM(g_i)
        # g_i = g_i.reshape([g_i.shape[1], -1])
        g_i, _ = self.job_biLSTM(g_i)
        g_i = torch.sum(g_i, dim=1)
        g_i = self.job_layer(g_i)
        return g_i

    def _forward_I(self, user_id, item_id, his_items, his_users, his_items_label, his_users_label):
        f_i = self._get_fg_I_user(user_id, his_items, his_items_label)
        g_i = self._get_fg_I_item(item_id, his_users, his_users_label)
        score_I = torch.mul(f_i, g_i).sum(dim=1)
        return score_I

    def calculate_loss(self, interaction):
        user_id = interaction[self.USER_ID]
        item_id = interaction[self.ITEM_ID]
        neg_item_id = interaction[self.NEG_ITEM_ID]

        his_items = interaction[self.HIS_ITEMS]
        his_items_label = interaction[self.HIS_ITEMS_LABEL]
        his_users = interaction[self.HIS_USERS]
        his_users_label = interaction[self.HIS_USERS_LABEL]
        neg_his_users = interaction[self.NEG_HIS_USERS]
        neg_his_users_label = interaction[self.NEG_HIS_USERS_LABEL]

        score_E = self._forward_E(user_id, item_id)
        score_E_neg = self._forward_E(user_id, neg_item_id)

        score_I = self._forward_I(user_id, item_id, his_items, his_users, his_items_label, his_users_label)
        score_I_neg = self._forward_I(user_id, neg_item_id, his_items, neg_his_users, his_items_label, neg_his_users_label)

        loss_E = self.loss(score_E, score_E_neg)
        loss_I = self.loss(score_I, score_I_neg)
        return loss_E + loss_I

    def predict(self, interaction):
        user_id = interaction[self.USER_ID]
        item_id = interaction[self.ITEM_ID]
        his_items = interaction[self.HIS_ITEMS]
        his_items_label = interaction[self.HIS_ITEMS_LABEL]

        his_users = interaction[self.HIS_USERS]
        his_users_label = interaction[self.HIS_USERS_LABEL]

        score_E = self._forward_E(user_id, item_id)
        score_I = self._forward_I(user_id, item_id, his_items, his_users, his_items_label, his_users_label)
        return self.sigmoid(score_E + score_I)
