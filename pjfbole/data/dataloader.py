# @Time   : 2022/3/4
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

import math
import copy
from logging import getLogger

import torch

from recbole.data.dataloader.general_dataloader import TrainDataLoader, NegSampleEvalDataLoader
from recbole.data.interaction import Interaction


class DualTrainDataloader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)


class DualNegDataloader(NegSampleEvalDataLoader):
    pass
    # def _neg_sampling(self, inter_feat):
    #     if 'dynamic' in self.neg_sample_args.keys() and self.neg_sample_args['dynamic'] != 'none':
    #         candidate_num = self.neg_sample_args['dynamic']
    #         user_ids = inter_feat[self.uid_field].numpy()
    #         item_ids = inter_feat[self.iid_field].numpy()
    #         neg_candidate_ids = self.sampler.sample_by_user_ids(user_ids, item_ids, self.neg_sample_num * candidate_num)
    #         self.model.eval()
    #         interaction = copy.deepcopy(inter_feat).to(self.model.device)
    #         interaction = interaction.repeat(self.neg_sample_num * candidate_num)
    #         neg_item_feat = Interaction({self.iid_field: neg_candidate_ids.to(self.model.device)})
    #         interaction.update(neg_item_feat)
    #         scores = self.model.predict(interaction).reshape(candidate_num, -1)
    #         indices = torch.max(scores, dim=0)[1].detach()
    #         neg_candidate_ids = neg_candidate_ids.reshape(candidate_num, -1)
    #         neg_item_ids = neg_candidate_ids[indices, [i for i in range(neg_candidate_ids.shape[1])]].view(-1)
    #         self.model.train()
    #         return self.sampling_func(inter_feat, neg_item_ids)
    #     elif self.neg_sample_args['strategy'] == 'by':
    #         user_ids = inter_feat[self.uid_field].numpy()
    #         item_ids = inter_feat[self.iid_field].numpy()
    #         neg_item_ids = self.sampler.sample_by_user_ids(user_ids, item_ids, self.neg_sample_num)
    #         return self.sampling_func(inter_feat, neg_item_ids)
    #     else:
    #         return inter_feat
