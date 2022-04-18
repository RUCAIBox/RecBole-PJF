# @Time   : 2022/4/18
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
from torch.nn.init import normal_

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType

# from model.layer import FusionLayer, HingeLoss


class IPJF(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, pool):
        super(IPJF, self).__init__(config, pool)
        self.NEG_USER_ID = config['NEG_PREFIX'] + self.USER_ID
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.neg_prefix = config['NEG_PREFIX']

        self.config = config
        self.pool = pool
        self.embedding_size = config['embedding_size']

        self.geek_fusion_layer = FusionLayer(self.embedding_size)
        self.job_fusion_layer = FusionLayer(self.embedding_size)

        self.w_job = nn.Linear(2 * self.embedding_size, 1)
        self.w_geek = nn.Linear(2 * self.embedding_size, 1)

        self.matching_mlp = nn.Sequential(
            nn.Linear(4 * self.embedding_size, 2 * self.embedding_size),
            nn.ReLU(),
            nn.Linear(2 * self.embedding_size, 1)
        )

        self.bert_lr = nn.Linear(config['BERT_embedding_size'], self.embedding_size)
        self.bert_user = dataset.bert_user.to(config['device'])
        self.bert_item = dataset.bert_item.to(config['device'])
        self.sigmoid = nn.Sigmoid()
        self.loss = HingeLoss()

        self.apply(self._init_weights)

    def forward(self, geek_id, job_id):
        # bert
        geek_vec = self.bert_lr(self.bert_user[geek_id])
        job_vec = self.bert_lr(self.bert_job[job_id])

        f_s = self.geek_fusion_layer(geek_vec, job_vec)
        f_e = self.job_fusion_layer(geek_vec, job_vec)
        r_s = self.w_geek(f_s)  # [2048, 1]
        r_e = self.w_job(f_e)  # [2048, 1]

        r_m = self.matching_mlp(torch.cat((f_s, f_e), dim=1))
        return r_s, r_e, r_m

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        if isinstance(module, nn.Parameter):
            normal_(module.weight.data, 0, 0.01)

    def calculate_geek_loss(self, geek_sents, job_sents, neu_job, neg_job):
        r_s_pos, _, r_m_pos = self.forward(geek_sents, job_sents)
        r_s, _, r_m = self.forward(geek_sents, neu_job)
        r_s_neg, _, r_m_neg = self.forward(geek_sents, neg_job)
        loss_s_i = self.loss(r_s_pos, r_s_neg) + self.loss(r_s, r_s_neg)  # geek intention
        loss_s_m = self.loss(r_m_pos, r_m) + self.loss(r_m_pos, r_m_neg)  # geek match
        return loss_s_i, loss_s_m

    def calculate_job_loss(self, geek_sents, job_sents, neu_geek, neg_geek):
        _, r_e_pos, r_m_pos = self.forward(geek_sents, job_sents)
        _, r_e, r_m = self.forward(neu_geek, job_sents)
        _, r_e_neg, r_m_neg = self.forward(neg_geek, job_sents)
        loss_e_i = self.loss(r_e_pos, r_e_neg) + self.loss(r_e, r_e_neg)  # job intention
        loss_e_m = self.loss(r_m_pos, r_m) + self.loss(r_m_pos, r_m_neg)  # job match
        return loss_e_i, loss_e_m

    def calculate_loss(self, interaction):
        geek_sents = interaction[self.USER_ID]
        job_sents = interaction[self.ITEM_ID]
        neu_job = interaction['neu_job_id']  # 中性岗位
        neg_job = interaction[self.NEG_JOB_ID]  # 负岗位
        neu_geek = interaction['neu_geek_id']  # 中性用户
        neg_geek = interaction[self.NEG_JOB_ID]  # 负用户

        loss_s_i, loss_s_m = self.calculate_geek_loss(geek_sents, job_sents, neu_job, neg_job)
        loss_e_i, loss_e_m = self.calculate_job_loss(geek_sents, job_sents, neu_geek, neg_geek)

        loss = loss_s_i + loss_s_m + loss_e_i + loss_e_m
        return loss

    def predict(self, interaction):
        geek_id = interaction[self.USER_ID]
        job_id = interaction[self.ITEM_ID]
        _, _, match_score = self.forward(geek_id, job_id)
        return torch.sigmoid(match_score)
