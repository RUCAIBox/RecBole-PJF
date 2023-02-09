# @Time   : 2022/3/4
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
pjfbole.data.dataloader
########################
"""

import torch
import numpy as np
from recbole.data.dataloader.general_dataloader import TrainDataLoader
from recbole.data.interaction import Interaction, cat_interactions

from logging import getLogger
from recbole.data.dataloader.abstract_dataloader import (
    AbstractDataLoader,
    NegSampleDataLoader,
)
from recbole.utils import InputType, ModelType

class RNegSampleEvalDataLoader(NegSampleDataLoader):
    """:class:`NegSampleEvalDataLoader` is a dataloader for neg-sampling evaluation.
    It is similar to :class:`TrainDataLoader` which can generate negative items,
    and this dataloader also permits that all the interactions corresponding to each user are in the same batch
    and positive interactions are before negative interactions.
    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self._set_neg_sample_args(
            config, dataset, InputType.POINTWISE, config["eval_neg_sample_args"]
        )
        if (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            user_num = dataset.user_num
            dataset.sort(by=dataset.uid_field, ascending=True)
            self.uid_list = []
            start, end = dict(), dict()
            for i, uid in enumerate(dataset.inter_feat[dataset.uid_field].numpy()):
                if uid not in start:
                    self.uid_list.append(uid)  # 所有出现的用户
                    start[uid] = i  # 用户 uid 在 inter_feat 第一次出现的位置
                end[uid] = i   # 用户 uid 在 inter_feat 最后一次出现的位置
            self.uid2index = np.array([None] * user_num)
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            for uid in self.uid_list:
                self.uid2index[uid] = slice(start[uid], end[uid] + 1)
                self.uid2items_num[uid] = end[uid] - start[uid] + 1
            self.uid_list = np.array(self.uid_list)
            self.sample_size = len(self.uid_list)
        else:
            self.sample_size = len(dataset)
        if shuffle:
            self.logger.warnning("NegSampleEvalDataLoader can't shuffle")
            shuffle = False
        
        self.bilateral_type = False
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        if (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            inters_num = sorted(self.uid2items_num * self.times, reverse=True)
            batch_num = 1
            new_batch_size = inters_num[0]
            for i in range(1, len(inters_num)):
                if new_batch_size + inters_num[i] > batch_size:
                    break
                batch_num = i + 1
                new_batch_size += inters_num[i]
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        self._set_neg_sample_args(
            config, self._dataset, InputType.POINTWISE, config["eval_neg_sample_args"]
        )
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        if (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            uid_list = self.uid_list[index]
            data_list = []
            idx_list = []
            positive_u = []
            positive_i = torch.tensor([], dtype=torch.int64)

            positive_u_list = []

            for idx, uid in enumerate(uid_list):
                index = self.uid2index[uid]
                transformed_data = self.transform(self._dataset, self._dataset[index])
                data_list.append(self._neg_sampling(transformed_data))
                idx_list += [idx for i in range(self.uid2items_num[uid] * self.times)]
                positive_u += [idx for i in range(self.uid2items_num[uid])]
                positive_i = torch.cat(
                    (positive_i, self._dataset[index][self.iid_field]), 0
                )

                positive_u_list += [uid]

            cur_data = cat_interactions(data_list)
            idx_list = torch.from_numpy(np.array(idx_list)).long()
            positive_u = torch.from_numpy(np.array(positive_u)).long()
            if self.bilateral_type == True:
                return cur_data, idx_list, positive_u, positive_i, positive_u_list
                
            return cur_data, idx_list, positive_u, positive_i
        else:
            data = self._dataset[index]
            transformed_data = self.transform(self._dataset, data)
            cur_data = self._neg_sampling(transformed_data)
            return cur_data, None, None, None


class RFullSortEvalDataLoader(AbstractDataLoader):
    """:class:`FullSortEvalDataLoader` is a dataloader for full-sort evaluation. In order to speed up calculation,
    this dataloader would only return then user part of interactions, positive items and used items.
    It would not return negative items.
    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.is_sequential = config["MODEL_TYPE"] == ModelType.SEQUENTIAL
        if not self.is_sequential:
            user_num = dataset.user_num
            self.uid_list = []
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            self.uid2positive_item = np.array([None] * user_num)
            self.uid2history_item = np.array([None] * user_num)

            dataset.sort(by=self.uid_field, ascending=True)
            last_uid = None
            positive_item = set()
            uid2used_item = sampler.used_ids
            for uid, iid in zip(
                dataset.inter_feat[self.uid_field].numpy(),
                dataset.inter_feat[self.iid_field].numpy(),
            ):
                if uid != last_uid:
                    self._set_user_property(
                        last_uid, uid2used_item[last_uid], positive_item
                    )
                    last_uid = uid
                    self.uid_list.append(uid)
                    positive_item = set()
                positive_item.add(iid)
            self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
            self.uid_list = torch.tensor(self.uid_list, dtype=torch.int64)
            self.user_df = dataset.join(Interaction({self.uid_field: self.uid_list}))

        self.sample_size = len(self.user_df) if not self.is_sequential else len(dataset)
        if shuffle:
            self.logger.warnning("FullSortEvalDataLoader can't shuffle")
            shuffle = False
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _set_user_property(self, uid, used_item, positive_item):
        if uid is None:
            return
        history_item = used_item - positive_item
        self.uid2positive_item[uid] = torch.tensor(
            list(positive_item), dtype=torch.int64
        )
        self.uid2items_num[uid] = len(positive_item)
        self.uid2history_item[uid] = torch.tensor(list(history_item), dtype=torch.int64)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        if not self.is_sequential:
            batch_num = max(batch_size // self._dataset.item_num, 1)
            new_batch_size = batch_num * self._dataset.item_num
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def collate_fn(self, index):
        index = np.array(index)
        if not self.is_sequential:
            user_df = self.user_df[index]
            uid_list = list(user_df[self.uid_field])

            history_item = self.uid2history_item[uid_list]
            positive_item = self.uid2positive_item[uid_list]

            history_u = torch.cat(
                [
                    torch.full_like(hist_iid, i)
                    for i, hist_iid in enumerate(history_item)
                ]
            )
            history_i = torch.cat(list(history_item))

            positive_u = torch.cat(
                [torch.full_like(pos_iid, i) for i, pos_iid in enumerate(positive_item)]
            )
            positive_i = torch.cat(list(positive_item))

            return user_df, (history_u, history_i), positive_u, positive_i, uid_list

            if self.bilateral_type == True:
                return user_df, (history_u, history_i), positive_u, positive_i, uid_list
                
            return user_df, (history_u, history_i), positive_u, positive_i
        else:
            interaction = self._dataset[index]
            transformed_interaction = self.transform(self._dataset, interaction)
            inter_num = len(transformed_interaction)
            positive_u = torch.arange(inter_num)
            positive_i = transformed_interaction[self.iid_field]

            return transformed_interaction, None, positive_u, positive_i


class PJFTrainDataLoader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self.used_ids_neg_i = self._sampler.used_ids
        self.change_field_direction()
        self._sampler.used_ids = self._sampler.get_used_ids()
        self._sampler = self._sampler.set_phase('train')

        self.used_ids_neg_u = self._sampler.used_ids
        self.change_field_direction()
        self._sampler.used_ids = self.used_ids_neg_i

    def change_field_direction(self):
        self.uid_field, self.iid_field = self.iid_field, self.uid_field
        self._sampler.uid_field, self._sampler.iid_field = self._sampler.iid_field, self._sampler.uid_field
        self._sampler.user_num, self._sampler.item_num = self._sampler.item_num, self._sampler.user_num

    def _neg_sampling(self, inter_feat):
        inter_feat_neg_i = super(PJFTrainDataLoader, self)._neg_sampling(inter_feat)

        self.change_field_direction()
        self._sampler.used_ids = self.used_ids_neg_u
        inter_feat_neg_u = super(PJFTrainDataLoader, self)._neg_sampling(inter_feat)

        self.change_field_direction()
        self._sampler.used_ids = self.used_ids_neg_i
        
        inter_feat.update(inter_feat_neg_i)
        inter_feat.update(inter_feat_neg_u)
        return inter_feat


class IPJFTrainDataLoader(PJFTrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        if dataset.label_field in dataset.user_single_inter.columns:
            self.user_inter = dataset.user_single_inter[dataset.user_single_inter[dataset.label_field] == 0]
            self.item_inter = dataset.item_single_inter[dataset.item_single_inter[dataset.label_field] == 0]
        else:
            self.user_inter = dataset.user_single_inter
            self.item_inter = dataset.item_single_inter
        self.neu_prefix = 'neu_'

    def change_direction(self):
        self.uid_field, self.iid_field = self.iid_field, self.uid_field
        self._sampler.uid_field, self._sampler.iid_field = self._sampler.iid_field, self._sampler.uid_field
        self._sampler.user_num, self._sampler.item_num = self._sampler.item_num, self._sampler.user_num
        self._sampler.used_ids = self._sampler.get_used_ids()
        self._sampler = self._sampler.set_phase('train')

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
        self.used_item_id = [[] for _ in range(self._sampler.user_num)]
        for uid, iid in zip(self.user_inter[self.uid_field].to_numpy(), self.user_inter[self.iid_field].to_numpy()):
            self.used_item_id[uid].append(iid)
        self.used_item_id = np.array(self.used_item_id)

        self.used_user_id = [[] for _ in range(self._sampler.item_num)]
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

        neu_item_feat = self.neu_sample_by_ids(user_ids, self.used_item_id, self._sampler.item_num, self.iid_field)
        neu_user_feat = self.neu_sample_by_ids(item_ids, self.used_user_id, self._sampler.user_num, self.uid_field)

        neu_item_feat.update(neu_user_feat)
        return neu_item_feat

    def _next_batch_data(self):
        cur_data = self._neg_sampling(self.dataset[self.pr:self.pr + self.step])
        self.pr += self.step
        return cur_data




class CausCTrainDataloader(PJFTrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        if dataset.label_field in dataset.user_single_inter.columns:
            self.user_inter = dataset.user_single_inter[dataset.user_single_inter[dataset.label_field] == 0]
            self.item_inter = dataset.item_single_inter[dataset.item_single_inter[dataset.label_field] == 0]
        else:
            self.user_inter = dataset.user_single_inter
            self.item_inter = dataset.item_single_inter
        self.neu_prefix = 'neu_'

    def change_direction(self):
        self.uid_field, self.iid_field = self.iid_field, self.uid_field
        self._sampler.uid_field, self._sampler.iid_field = self._sampler.iid_field, self._sampler.uid_field
        self._sampler.user_num, self._sampler.item_num = self._sampler.item_num, self._sampler.user_num
        self._sampler.used_ids = self._sampler.get_used_ids()
        self._sampler = self._sampler.set_phase('train')

    def _neg_sampling(self, inter_feat):
        inter_feat_neg_i = super(CausCFTrainDataloader, self)._neg_sampling(inter_feat)

        self.change_direction()
        inter_feat_neg_u = super(CausCFTrainDataloader, self)._neg_sampling(inter_feat)
        self.change_direction()

        inter_feat_nue = self._neu_sampling(inter_feat)

        inter_feat.update(inter_feat_neg_i)
        inter_feat.update(inter_feat_neg_u)
        inter_feat.update(inter_feat_nue)
        return inter_feat

    def get_neu_ids(self):
        # get used neu ids for sampling
        self.used_item_id = [[] for _ in range(self._sampler.user_num)]
        for uid, iid in zip(self.user_inter[self.uid_field].to_numpy(), self.user_inter[self.iid_field].to_numpy()):
            self.used_item_id[uid].append(iid)
        self.used_item_id = np.array(self.used_item_id)

        self.used_user_id = [[] for _ in range(self._sampler.item_num)]
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

        neu_item_feat = self.neu_sample_by_ids(user_ids, self.used_item_id, self._sampler.item_num, self.iid_field)
        neu_user_feat = self.neu_sample_by_ids(item_ids, self.used_user_id, self._sampler.user_num, self.uid_field)

        neu_item_feat.update(neu_user_feat)
        return neu_item_feat
