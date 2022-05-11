# @Time   : 2022/3/4
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
pjfbole.data.dataloader
########################
"""

import torch
import numpy as np
from recbole.data.dataloader.general_dataloader import TrainDataLoader, NegSampleEvalDataLoader
from recbole.data.interaction import Interaction


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


class IPJFTrainDataloader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self.user_inter = dataset.user_single_inter[dataset.user_single_inter[dataset.label_field] == 0]
        self.item_inter = dataset.item_single_inter[dataset.item_single_inter[dataset.label_field] == 0]
        self.neu_prefix = 'neu_'

    def change_direction(self):
        self.uid_field, self.iid_field = self.iid_field, self.uid_field
        self.sampler.uid_field, self.sampler.iid_field = self.sampler.iid_field, self.sampler.uid_field
        self.sampler.user_num, self.sampler.item_num = self.sampler.item_num, self.sampler.user_num
        self.sampler.used_ids = self.sampler.get_used_ids()
        self.sampler = self.sampler.set_phase('train')

    def _neg_sampling(self, inter_feat):
        inter_feat_neg_i = super(IPJFTrainDataloader, self)._neg_sampling(inter_feat)

        self.change_direction()
        inter_feat_neg_u = super(IPJFTrainDataloader, self)._neg_sampling(inter_feat)
        self.change_direction()

        inter_feat_nue = self._neu_sampling(inter_feat)

        inter_feat.update(inter_feat_neg_i)
        inter_feat.update(inter_feat_neg_u)
        inter_feat.update(inter_feat_nue)
        return inter_feat

    def get_neu_ids(self):
        # get used neu ids for sampling
        self.used_item_id = [[] for _ in range(self.sampler.user_num)]
        for uid, iid in zip(self.user_inter[self.uid_field].to_numpy(), self.user_inter[self.iid_field].to_numpy()):
            self.used_item_id[uid].append(iid)
        self.used_item_id = np.array(self.used_item_id)

        self.used_user_id = [[] for _ in range(self.sampler.item_num)]
        for uid, iid in zip(self.item_inter[self.uid_field].to_numpy(), self.item_inter[self.iid_field].to_numpy()):
            self.used_user_id[iid].append(uid)
        self.used_user_id = np.array(self.used_user_id)

    def neu_sample_by_ids(self, key_ids, used_ids, num, field):
        # neu sample by ids
        neu_num = len(key_ids)
        value_ids = np.zeros(neu_num, dtype=np.int64)
        check_list = np.arange(neu_num)
        value_ids[check_list] = np.random.randint(1, num, len(check_list))
        value_ids = [value_ids[i] if len(used_ids[key_ids[i]]) == 0
                     else used_ids[key_ids[i]][value_ids[i] % len(used_ids[key_ids[i]])]
                     for i in check_list]
        neu_ids = torch.tensor(value_ids)
        neu_feat = Interaction({field: neu_ids})
        neu_feat = self.dataset.join(neu_feat)
        neu_feat.add_prefix(self.neu_prefix)
        return neu_feat

    def _neu_sampling(self, inter_feat):
        self.get_neu_ids()
        user_ids = inter_feat[self.uid_field].numpy()
        item_ids = inter_feat[self.iid_field].numpy()

        neu_item_feat = self.neu_sample_by_ids(user_ids, self.used_item_id, self.sampler.item_num, self.iid_field)
        neu_user_feat = self.neu_sample_by_ids(item_ids, self.used_user_id, self.sampler.user_num, self.uid_field)

        neu_item_feat.update(neu_user_feat)
        return neu_item_feat

    def _next_batch_data(self):
        cur_data = self._neg_sampling(self.dataset[self.pr:self.pr + self.step])
        self.pr += self.step
        return cur_data


class LFRRTrainDataloader(DPGNNTrainDataloader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)


class PJFFFTrainDataloader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _neg_sampling(self, inter_feat):
        inter_feat_neg = super(PJFFFTrainDataloader, self)._neg_sampling(inter_feat)
        inter_feat.update(inter_feat_neg)
        # inter_feat = pd.merge(inter_feat, dataset.his_user, on='item_id')
        # inter_feat = pd.merge(inter_feat, dataset.his_item, on='user_id')
        # inter_feat = pd.merge(inter_feat, dataset.his_user, on='item_id')
        return inter_feat
