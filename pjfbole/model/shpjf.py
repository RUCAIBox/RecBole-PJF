# @Time   : 2022/5/5
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
SHPJF
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.layers import MLPLayers
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class SHPJF(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SHPJF, self).__init__(config, dataset)

        self.ITEM_SENTS = config['ITEM_DOC_FIELD']
        self.neg_prefix = config['NEG_PREFIX']

        self.wd_embedding_size = config['wd_embedding_size']
        self.user_embedding_size = config['user_embedding_size']
        self.bert_embedding_size = config['bert_embedding_size']
        self.hd_size = config['hidden_size']
        self.dropout = config['dropout']
        self.num_heads = config['num_heads']
        self.query_wd_len = config['query_wd_len']
        self.query_his_len = config['query_his_len']
        self.max_job_longsent_len = config['max_longsent_len']
        self.beta = config['beta']
        self.k = config['k']
        self.wd_num = len(dataset.wd2id.keys())

        self.emb = nn.Embedding(self.wd_num, self.wd_embedding_size, padding_idx=0)
        self.geek_emb = nn.Embedding(self.n_users, self.user_embedding_size, padding_idx=0)
        nn.init.xavier_normal_(self.geek_emb.weight.data)
        self.job_emb = nn.Embedding(self.n_items, self.user_embedding_size, padding_idx=0)
        nn.init.xavier_normal_(self.job_emb.weight.data)

        self.text_matching_fc = nn.Linear(self.bert_embedding_size, self.hd_size)

        self.pos_enc = nn.Parameter(torch.rand(1, self.query_his_len, self.user_embedding_size))
        self.q_pos_enc = nn.Parameter(torch.rand(1, self.query_his_len, self.user_embedding_size))

        self.job_desc_attn_layer = nn.Linear(self.wd_embedding_size, 1)

        self.wq = nn.Linear(self.wd_embedding_size, self.user_embedding_size, bias=False)
        self.text_based_lfc = nn.Linear(self.query_his_len, self.k, bias=False)
        self.job_emb_lfc = nn.Linear(self.query_his_len, self.k, bias=False)

        self.text_based_attn_layer = nn.MultiheadAttention(
            embed_dim=self.user_embedding_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            bias=False
        )
        self.text_based_im_fc = nn.Linear(self.user_embedding_size, self.user_embedding_size)

        self.job_emb_attn_layer = nn.MultiheadAttention(
            embed_dim=self.user_embedding_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            bias=False
        )
        self.job_emb_im_fc = nn.Linear(self.user_embedding_size, self.user_embedding_size)

        self.intent_fusion = MLPLayers(
            layers=[self.user_embedding_size * 4, self.hd_size, 1],
            dropout=self.dropout,
            activation='tanh'
        )

        self.pre_mlp = MLPLayers(
            layers=[self.hd_size + 2, self.hd_size, 1],
            dropout=self.dropout,
            activation='tanh'
        )

        self.sigmoid = nn.Sigmoid()
        self.loss = BPRLoss()

    def _text_matching_layer(self, inter_bert_vec):
        x = self.text_matching_fc(inter_bert_vec)   # (B, wordD)
        return x

    def _intent_modeling_layer(self, job_id, job_longsent, job_his, qwd_his, qlen_his):
        job_longsent_len = torch.sum(job_longsent != 0, dim=-1, keepdim=True)
        job_desc_vec = self.emb(job_longsent)                   # (B, L, wordD)
        job_desc_mask = torch.arange(self.max_job_longsent_len, device=job_desc_vec.device) \
                           .expand(len(job_longsent_len), self.max_job_longsent_len) \
                           >= job_longsent_len
        job_desc_attn_weight = self.job_desc_attn_layer(job_desc_vec)
        job_desc_attn_weight = torch.masked_fill(job_desc_attn_weight, job_desc_mask.unsqueeze(-1), -10000)
        job_desc_attn_weight = torch.softmax(job_desc_attn_weight, dim=1)
        job_desc_vec = torch.sum(job_desc_attn_weight * job_desc_vec, dim=1)
        job_desc_vec = self.wq(job_desc_vec)                    # (B, idD)

        job_id_vec = self.job_emb(job_id)                       # (B, idD)

        job_his_vec = self.job_emb(job_his)                     # (B, Q, idD)
        job_his_vec = job_his_vec + self.pos_enc

        qwd_his_vec = self.emb(qwd_his)                         # (B, Q, W, wordD)
        qlen_his = torch.where(qlen_his < 1, torch.ones(1, device=qlen_his.device, dtype=qlen_his.dtype), qlen_his)
        qwd_his_vec = torch.sum(qwd_his_vec, dim=2) / \
                      qlen_his.unsqueeze(-1)                    # (B, Q, wordD)
        qwd_his_vec = self.wq(qwd_his_vec)                      # (B, Q, idD)
        qwd_his_vec = self.q_pos_enc + qwd_his_vec

        proj_qwd_his_vec = self.text_based_lfc(qwd_his_vec.transpose(2, 1)).transpose(2, 1) * self.k / self.query_his_len
                                                                # (B, K, idD)
        proj_job_his_vec = self.job_emb_lfc(job_his_vec.transpose(2, 1)).transpose(2, 1) * self.k / self.query_his_len
                                                                # (B, K, idD)
        text_based_intent_vec, _ = self.text_based_attn_layer(
            query=job_desc_vec.unsqueeze(0),
            key=proj_qwd_his_vec.transpose(1, 0),
            value=proj_job_his_vec.transpose(1, 0)
        )
        text_based_intent_vec = text_based_intent_vec.squeeze(0)# (B, idD)
        text_based_intent_vec = self.text_based_im_fc(text_based_intent_vec)

        job_emb_intent_vec, _ = self.job_emb_attn_layer(
            query=job_id_vec.unsqueeze(0),
            key=proj_job_his_vec.transpose(1, 0),
            value=proj_job_his_vec.transpose(1, 0),
        )
        job_emb_intent_vec = job_emb_intent_vec.squeeze(0)      # (B, idD)
        job_emb_intent_vec = self.job_emb_im_fc(job_emb_intent_vec)

        intent_vec = (1 - self.beta) * text_based_intent_vec + self.beta * job_emb_intent_vec

        intent_modeling_vec = self.intent_fusion(
            torch.cat(
                [job_id_vec, intent_vec, job_id_vec - intent_vec, job_id_vec * intent_vec]
            , dim=1)
        )

        return intent_modeling_vec

    def _mf_layer(self, geek_id, job_id):
        geek_vec = self.geek_emb(geek_id)
        job_vec = self.job_emb(job_id)
        x = torch.sum(torch.mul(geek_vec, job_vec), dim=1, keepdim=True)
        return x

    def predict_layer(self, vecs):
        x = torch.cat(vecs, dim=-1)
        score = self.pre_mlp(x).squeeze(-1)
        return score

    def forward(self, interaction, neg_sample=False):
        inter_bert_vec = interaction['bert']
        text_matching_vec = self._text_matching_layer(inter_bert_vec)

        if not neg_sample:
            job_id = interaction[self.ITEM_ID]
            job_longsent = interaction['long_' + self.ITEM_SENTS]
        else:
            job_id = interaction[self.neg_prefix + self.ITEM_ID]
            job_longsent = interaction[self.neg_prefix + 'long_' + self.ITEM_SENTS]
        job_his = interaction['job_his']                            # (B, Q)
        qwd_his = interaction['qwd_his']                            # (B, Q * W)
        qwd_his = qwd_his.reshape(job_his.shape[0], job_his.shape[1], self.query_wd_len)
        job_his = job_his[:,:self.query_his_len]
        qwd_his = qwd_his[:,:self.query_his_len,:]
        qlen_his = interaction['qlen_his'][:,:self.query_his_len]   # (B, Q)
        intent_modeling_vec = self._intent_modeling_layer(job_id, job_longsent, job_his, qwd_his, qlen_his)

        geek_id = interaction[self.USER_ID]
        mf_vec = self._mf_layer(geek_id, job_id)
        score = self.predict_layer([text_matching_vec, intent_modeling_vec, mf_vec])
        return score

    def calculate_loss(self, interaction):
        output = self.forward(interaction)
        output_neg = self.forward(interaction, neg_sample=True)
        return self.loss(output, output_neg)

    def predict(self, interaction):
        score = self.forward(interaction)
        return self.sigmoid(score)
