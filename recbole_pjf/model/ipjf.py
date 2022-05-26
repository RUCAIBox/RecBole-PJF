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


class IPJF(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(IPJF, self).__init__(config, dataset)
        self.embedding_size = config['embedding_size']
        self.is_add_bert = config['ADD_BERT']

        if not self.is_add_bert:
            self.USER = config['USER_DOC_FIELD']
            self.ITEM = config['ITEM_DOC_FIELD']

            self.NEU_USER = 'neu_' + self.USER
            self.NEU_ITEM = 'neu_' + self.ITEM

            self.NEG_USER = config['NEG_PREFIX'] + self.USER
            self.NEG_ITEM = config['NEG_PREFIX'] + self.ITEM

            self.emb = nn.Embedding(len(dataset.wd2id.keys()), self.embedding_size, padding_idx=0)
            self.geek_channels = config['max_sent_num']  # 10
            self.job_channels = config['max_sent_num']  # 10
            self.geek_layer = DocumentEncoder(
                channels=self.geek_channels,
                kernel_size=[(5, 1), (3, 1)],
                pool_size=(2, 1),
                dim=self.embedding_size,
                method='max'
            )
            self.job_layer = DocumentEncoder(
                channels=self.job_channels,
                kernel_size=[(5, 1), (5, 1)],
                pool_size=(2, 1),
                dim=self.embedding_size,
                method='mean'
            )
            self.w_att = nn.Parameter(torch.rand(self.embedding_size, self.embedding_size))
            self.register_parameter('w_att', self.w_att)

            self.tanh = nn.Tanh()
        else:
            self.USER = self.USER_ID
            self.ITEM = self.ITEM_ID

            self.NEU_USER = 'neu_' + self.USER
            self.NEU_ITEM = 'neu_' + self.ITEM

            self.NEG_USER = config['NEG_PREFIX'] + self.USER
            self.NEG_ITEM = config['NEG_PREFIX'] + self.ITEM

            self.bert_lr = nn.Linear(config['BERT_embedding_size'], self.embedding_size)
            self.bert_user = dataset.bert_user.to(config['device'])
            self.bert_item = dataset.bert_item.to(config['device'])

        self.geek_fusion_layer = FusionLayer(self.embedding_size)
        self.job_fusion_layer = FusionLayer(self.embedding_size)
        self.w_job = nn.Linear(2 * self.embedding_size, 1)
        self.w_geek = nn.Linear(2 * self.embedding_size, 1)

        self.matching_mlp = nn.Sequential(
            nn.Linear(4 * self.embedding_size, 2 * self.embedding_size),
            nn.ReLU(),
            nn.Linear(2 * self.embedding_size, 1)
        )

        self.sigmoid = nn.Sigmoid()
        self.loss = HingeLoss()

        self.apply(self._init_weights)

    def doc_encode_by_cnn(self, geek_docs, job_docs):
        geek_docs = self.emb(geek_docs)
        job_docs = self.emb(job_docs)

        geek_matrix = self.geek_layer(geek_docs)  # [2048, 20, 64]
        job_matrix = self.job_layer(job_docs)  # [2048, 20, 64]

        # A = job_matrix \cdot self.w_att \cdot geek_matrix
        A = torch.matmul(geek_matrix, self.w_att)
        A = torch.matmul(A, job_matrix.permute(0, 2, 1))  # [2048, 20, 20]

        geek_attn = A.sum(dim=2)  # [2048, 20]
        geek_attn = geek_attn / torch.max(geek_attn)
        geek_attn = torch.softmax(geek_attn, dim=1)  # [2048, 20]
        geek_vec = torch.sum((geek_matrix.permute(2, 0, 1) * geek_attn).permute(1, 2, 0), dim=1)  # [2048, 64]

        job_attn = A.sum(dim=1)
        job_attn = job_attn / torch.max(job_attn)
        job_attn = torch.softmax(job_attn, dim=1)
        job_vec = torch.sum((job_matrix.permute(2, 0, 1) * job_attn).permute(1, 2, 0), dim=1)   # [2048, 64]
        return geek_vec, job_vec

    def doc_encode_by_bert(self, geek_id, job_id):
        geek_vec = self.bert_lr(self.bert_user[geek_id])
        job_vec = self.bert_lr(self.bert_item[job_id])
        return geek_vec, job_vec

    def forward(self, geek, job):
        if not self.is_add_bert:
            geek_vec, job_vec = self.doc_encode_by_cnn(geek, job)
        else:
            geek_vec, job_vec = self.doc_encode_by_bert(geek, job)

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

    def calculate_geek_loss(self, pos_geek, pos_job, neu_job, neg_job):
        r_s_pos, _, r_m_pos = self.forward(pos_geek, pos_job)
        r_s, _, r_m = self.forward(pos_geek, neu_job)
        r_s_neg, _, r_m_neg = self.forward(pos_geek, neg_job)
        loss_s_i = self.loss(r_s_pos, r_s_neg) + self.loss(r_s, r_s_neg)  # geek intention
        loss_s_m = self.loss(r_m_pos, r_m) + self.loss(r_m_pos, r_m_neg)  # geek match
        return loss_s_i, loss_s_m

    def calculate_job_loss(self, pos_geek, pos_job, neu_geek, neg_geek):
        _, r_e_pos, r_m_pos = self.forward(pos_geek, pos_job)
        _, r_e, r_m = self.forward(neu_geek, pos_job)
        _, r_e_neg, r_m_neg = self.forward(neg_geek, pos_job)
        loss_e_i = self.loss(r_e_pos, r_e_neg) + self.loss(r_e, r_e_neg)  # job intention
        loss_e_m = self.loss(r_m_pos, r_m) + self.loss(r_m_pos, r_m_neg)  # job match
        return loss_e_i, loss_e_m

    def calculate_loss(self, interaction):
        pos_geek = interaction[self.USER]
        pos_job = interaction[self.ITEM]

        neu_job = interaction[self.NEU_ITEM]
        neg_job = interaction[self.NEG_ITEM]

        neu_geek = interaction[self.NEU_USER]
        neg_geek = interaction[self.NEG_USER]

        loss_s_i, loss_s_m = self.calculate_geek_loss(pos_geek, pos_job, neu_job, neg_job)
        loss_e_i, loss_e_m = self.calculate_job_loss(pos_geek, pos_job, neu_geek, neg_geek)

        loss = loss_s_i + loss_s_m + loss_e_i + loss_e_m
        return loss

    def predict(self, interaction):
        geek_id = interaction[self.USER]
        job_id = interaction[self.ITEM]
        _, _, match_score = self.forward(geek_id, job_id)
        return torch.sigmoid(match_score).squeeze()


class DocumentEncoder(nn.Module):
    def __init__(self, channels, kernel_size, pool_size, dim, method='max'):
        super(DocumentEncoder, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size[0]),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size[1]),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, dim))
        )

    def forward(self, x):
        x = self.net1(x)  # [2048, 20, 30, 64]
        x = self.net2(x).squeeze(2)  # [2048, 20, 64]
        return x


class SimpleFusionLayer(nn.Module):
    def __init__(self, hd_size):
        super(SimpleFusionLayer, self).__init__()
        self.fc = nn.Linear(hd_size * 4, hd_size)

    def forward(self, a, b):
        assert a.shape == b.shape
        x = torch.cat([a, b, a * b, a - b], dim=-1)
        x = self.fc(x)
        x = torch.tanh(x)
        return x


class FusionLayer(nn.Module):
    def __init__(self, hd_size):
        super(FusionLayer, self).__init__()
        self.m = SimpleFusionLayer(hd_size)
        self.g = nn.Sequential(
            nn.Linear(hd_size * 2, 1),
            nn.Sigmoid()
        )

    def _single_layer(self, a, b):
        ma = self.m(a, b)
        x = torch.cat([a, b], dim=-1)
        ga = self.g(x)
        return ga * ma + (1 - ga) * a

    def forward(self, a, b):
        assert a.shape == b.shape
        a = self._single_layer(a, b)
        b = self._single_layer(b, a)
        return torch.cat([a, b], dim=-1)


class HingeLoss(torch.nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.delta = 0.05

    def forward(self, pos_score, neg_score):
        hinge_loss = torch.clamp(neg_score - pos_score+ self.delta, min=0).mean()
        return hinge_loss