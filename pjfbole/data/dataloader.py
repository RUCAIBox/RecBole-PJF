# @Time   : 2022/3/4
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
pjfbole.data.dataloader
########################
"""

import torch

from recbole.data.dataloader.general_dataloader import TrainDataLoader, NegSampleEvalDataLoader


class DPGNNTrainDataloader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def change_direction(self):
        self.uid_field, self.iid_field = self.iid_field, self.uid_field
        self.sampler.uid_field, self.sampler.iid_field = self.sampler.iid_field, self.sampler.uid_field
        self.sampler.user_num, self.sampler.item_num = self.sampler.item_num, self.sampler.user_num
        self.sampler.used_ids = self.sampler.get_used_ids()
        self.sampler = self.sampler.set_phase('train')

    def _neg_sampling(self, inter_feat):
        inter_feat_neg_i = super(DPGNNTrainDataloader, self)._neg_sampling(inter_feat)

        self.change_direction()
        inter_feat_neg_u = super(DPGNNTrainDataloader, self)._neg_sampling(inter_feat)
        self.change_direction()

        inter_feat.update(inter_feat_neg_i)
        inter_feat.update(inter_feat_neg_u)
        return inter_feat
