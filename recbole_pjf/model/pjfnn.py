# @Time   : 2022/3/4
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class TextCNN(nn.Module):
    def __init__(self, channels, kernel_size, pool_size, dim, method='max'):
        super(TextCNN, self).__init__()
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
        if method == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, dim))
        elif method == 'mean':
            self.pool = nn.AdaptiveAvgPool2d((1, dim))
        else:
            raise ValueError('method {} not exist'.format(method))

    def forward(self, x):
        # import pdb
        # pdb.set_trace()   # [2048, 20, 30, 64]
        x = self.net1(x)  # [2048, 20, 13, 64]
        x = self.net2(x).squeeze(2)  # [2048, 20, 64]
        x = self.pool(x).squeeze(1)  # [2048, 64]
        return x


class PJFNN(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(PJFNN, self).__init__(config, dataset)
        self.USER_SENTS = config['USER_DOC_FIELD']
        self.ITEM_SENTS = config['ITEM_DOC_FIELD']
        self.neg_prefix = config['NEG_PREFIX']
        # load parameters info
        self.embedding_size = config['embedding_size']
        self.geek_channels = config['max_sent_num']  # 10
        self.job_channels = config['max_sent_num']  # 10
        self.emb = nn.Embedding(len(dataset.wd2id.keys()), self.embedding_size, padding_idx=0)

        # define layers and loss
        self.geek_layer = TextCNN(
            channels=self.geek_channels,
            kernel_size=[(5, 1), (3, 1)],
            pool_size=(2, 1),
            dim=self.embedding_size,
            method='max'
        )

        self.job_layer = TextCNN(
            channels=self.job_channels,
            kernel_size=[(5, 1), (5, 1)],
            pool_size=(2, 1),
            dim=self.embedding_size,
            method='mean'
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, 1)
        )
        # self.loss = nn.BCEWithLogitsLoss()

        self.loss = BPRLoss()
        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, geek_sents, job_sents):
        geek_vec = self.emb(geek_sents)
        job_vec = self.emb(job_sents)
        geek_vec = self.geek_layer(geek_vec)
        job_vec = self.job_layer(job_vec)
        x = geek_vec * job_vec
        x = self.mlp(x).squeeze(1)
        return x

    def calculate_loss(self, interaction):
        geek_sents = interaction[self.USER_SENTS]
        job_sents = interaction[self.ITEM_SENTS]
        neg_job_sents = interaction[self.neg_prefix + self.ITEM_SENTS]

        output_pos = self.forward(geek_sents, job_sents)
        output_neg = self.forward(geek_sents, neg_job_sents)

        return self.loss(output_pos, output_neg)

    def predict(self, interaction):
        geek_sents = interaction[self.USER_SENTS]
        job_sents = interaction[self.ITEM_SENTS]
        return torch.sigmoid(self.forward(geek_sents, job_sents))

