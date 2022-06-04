# @Time   : 2022/03/01
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
recbole_pjf.data.pjf_dataset
##########################
"""

import copy
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from recbole.data.dataset import Dataset
from recbole.utils import FeatureType, set_color
from recbole.data.interaction import Interaction


class PJFDataset(Dataset):
    """:class:`PJFDataset` is inherited from :class:`recbole.data.dataset.Dataset`

    """
    def __init__(self, config):
        self.wd2id = {
            '[WD_PAD]': 0,
            '[WD_MISS]': 1
        }
        self.id2wd = ['[WD_PAD]', '[WD_MISS]']
        self.user_doc = None
        self.item_doc = None
        super().__init__(config)

    def change_direction(self):
        """Change direction for Validation and testing.
        """
        self.uid_field, self.iid_field = self.iid_field, self.uid_field
        self.user_feat, self.item_feat = self.item_feat, self.user_feat
        self.user_doc, self.item_doc = self.item_doc, self.user_doc

    def _change_feat_format(self):
        super(PJFDataset, self)._change_feat_format()
        if self.user_doc is not None:
            self._change_doc_format()

    def _data_processing(self):
        super(PJFDataset, self)._data_processing()
        if self.user_doc is not None:
            self._doc_fillna()

        if self.config['multi_direction']:
            self.direct_field = self.config['DIRECT_FIELD']
            if self.direct_field:
                self.geek_direct = self.field2token_id[self.direct_field]['0']
                self.user_single_inter = self.inter_feat[self.inter_feat[self.direct_field] == self.geek_direct]
                self.item_single_inter = self.inter_feat[self.inter_feat[self.direct_field] != self.geek_direct]
            else:
                self.user_single_inter = self.inter_feat
                self.item_single_inter = self.inter_feat

        self.inter_feat_all = self.inter_feat
        if self.label_field in self.inter_feat.columns:
            self.inter_feat = self.inter_feat[self.inter_feat[self.label_field] == 1]
        # is load pre-trained text vec
        if self.config['ADD_BERT'] and self.user_doc is not None:
            self._collect_text_vec()

    def _collect_text_vec(self):
        self.bert_user = torch.stack(self.user_doc[self.udoc_field + '_vec'].values.tolist(), dim=1).transpose(0, 1)
        self.bert_item = torch.stack(self.item_doc[self.idoc_field + '_vec'].values.tolist(), dim=1).transpose(0, 1)

    def _doc_fillna(self):
        def fill_nan(value, fill_size):
            if isinstance(value, (list, np.ndarray, torch.Tensor)):
                return value
            else:
                return torch.zeros(fill_size).long()

        ufield_list = [self.udoc_field, 'long_' + self.udoc_field, self.udoc_field + '_vec']
        ifield_list = [self.idoc_field, 'long_' + self.idoc_field, self.idoc_field + '_vec']
        doc_nan_size = [self.config['max_sent_num'], self.config['max_sent_len']]
        longdoc_nan_size = [1]
        vec_nan_size = [768]
        nan_size_list = [doc_nan_size, longdoc_nan_size, vec_nan_size]

        if self.user_doc is not None and self.item_doc is not None:
            new_udoc_df = pd.DataFrame({self.uid_field: np.arange(self.user_num)})
            self.user_doc = pd.merge(new_udoc_df, self.user_doc, on=self.uid_field, how='left')

            for field, nan_size in zip(ufield_list, nan_size_list):
                if field in self.user_doc.columns:
                    self.user_doc[field].fillna(value=0, inplace=True)
                    self.user_doc[field] = self.user_doc[field].apply(fill_nan, fill_size=nan_size)

            new_idoc_df = pd.DataFrame({self.iid_field: np.arange(self.item_num)})
            self.item_doc = pd.merge(new_idoc_df, self.item_doc, on=self.iid_field, how='left')

            for field, nan_size in zip(ifield_list, nan_size_list):
                if field in self.item_doc.columns:
                    self.item_doc[field].fillna(value=0, inplace=True)
                    self.item_doc[field] = self.item_doc[field].apply(fill_nan, fill_size=nan_size)

    def _change_doc_format(self):
        # 改变文本信息形式
        self.user_doc = self._doc_dataframe_to_interaction(self.user_doc)
        self.item_doc = self._doc_dataframe_to_interaction(self.item_doc)

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        datasets = super(PJFDataset, self).build()

        if self.config['multi_direction']:
            # valid_g = self.copy(datasets[1].inter_feat[datasets[1].inter_feat[self.direct_field] == self.geek_direct])
            valid_g = self.copy(datasets[1].inter_feat)

            # valid_j = self.copy(datasets[1].inter_feat[datasets[1].inter_feat[self.direct_field] != self.geek_direct])
            valid_j = self.copy(datasets[1].inter_feat)
            valid_j.change_direction()

            # test_g = self.copy(datasets[2].inter_feat[datasets[2].inter_feat[self.direct_field] == self.geek_direct])
            test_g = self.copy(datasets[2].inter_feat)

            # test_j = self.copy(datasets[2].inter_feat[datasets[2].inter_feat[self.direct_field] != self.geek_direct])
            test_j = self.copy(datasets[2].inter_feat)
            test_j.change_direction()
            return [datasets[0], valid_g, valid_j, test_g, test_j]
        return datasets

    def _get_field_from_config(self):
        """Initialization common field names.
        """
        super(PJFDataset, self)._get_field_from_config()
        self.udoc_field = self.config['USER_DOC_FIELD']
        self.idoc_field = self.config['ITEM_DOC_FIELD']
        self.logger.debug(set_color('udoc_field', 'blue') + f': {self.udoc_field}')
        self.logger.debug(set_color('idoc_field', 'blue') + f': {self.idoc_field}')

    def _load_data(self, token, dataset_path):
        """Load features of the resume and job description.

        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        super(PJFDataset, self)._load_data(token, dataset_path)
        if 'udoc' in self.config['load_col']:
            self.user_doc = self._load_user_or_item_doc(token, dataset_path, 'udoc', 'uid_field', 'udoc_field')
            self.item_doc = self._load_user_or_item_doc(token, dataset_path, 'idoc', 'iid_field', 'idoc_field')
            self.filter_data_with_no_doc()

    def _load_user_or_item_doc(self, token, dataset_path, suf, field_name, doc_field_name):
        """Load user/item doc.
        Returns:
            pandas.DataFrame: Loaded doc
        """
        feat_path = os.path.join(dataset_path, f'{token}.{suf}')
        field = getattr(self, field_name, None)
        doc_field = getattr(self, doc_field_name, None)

        if os.path.isfile(feat_path):
            feat = self._load_feat(feat_path, suf)
            self.logger.debug(f'[{suf}] feature loaded successfully from [{feat_path}].')
        else:
            feat = None
            self.logger.debug(f'[{feat_path}] not found, [{suf}] features are not loaded.')

        def word_map(sent):
            value = []
            for i, wd in enumerate(sent):
                if wd not in self.wd2id.keys():
                    self.wd2id[wd] = len(self.wd2id)
                    self.id2wd.append(wd)
                value.append(self.wd2id[wd])
            return value

        def get_long_doc(single_doc: list):
            long_doc = np.array([]).astype(int)
            for s in single_doc:
                long_doc = np.append(long_doc, s)
                if len(long_doc) > self.config['max_longsent_len']:
                    return long_doc[: self.config['max_longsent_len']]
            return long_doc

        def get_docs(single_doc: list):
            array_size = [self.config['max_sent_num'], self.config['max_sent_len']]
            docs = np.zeros(array_size).astype(int)
            sent_num = 0
            for s in single_doc:
                if len(s) > array_size[1]:
                    docs[sent_num] = s[:array_size[1]]
                else:
                    docs[sent_num] = np.pad(s, (0, array_size[1] - len(s)))  # doc[idx] 第idx个用户的多个句子组成的 tensor 矩阵
                sent_num += 1
                if sent_num >= array_size[0]:
                    break
            return docs

        if feat is not None and field is None:
            raise ValueError(f'{field_name} must be exist if {suf}_feat exist.')
        if feat is not None and field not in feat:
            raise ValueError(f'{field_name} must be loaded if {suf}_feat is loaded.')

        def bert_encoding(doc_list: list):
            s = ''
            for line in doc_list:
                for wd in line:
                    s += wd + ' '
            s = s[:512]
            input = self.tokenizer(s, return_tensors="pt")
            output = self.model(**input)[0][:, 0].detach()
            return output.squeeze()

        def deal_str(vec_string):
            s_res = list(map(lambda x: float(x), vec_string.split(',')))
            return torch.Tensor(s_res)

        if self.config['ADD_BERT']:
            # load bert vector
            bert_vec_save_path = os.path.join(dataset_path, f'{token}.{field_name[0]}vec')
            if os.path.isfile(bert_vec_save_path):
                vec = pd.read_csv(bert_vec_save_path, dtype=object)
                vec[doc_field + '_vec_str'] = vec[doc_field + '_vec_str'].apply(deal_str)
            else:
                # calculate and get bert vector
                from transformers import AutoTokenizer, AutoModel, logging
                logging.set_verbosity_warning()
                MODEL_PATH = self.config['pretrained_bert_model'] or "bert-base-cased"
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
                self.model = AutoModel.from_pretrained(MODEL_PATH)

                tqdm.pandas(desc=f"bert encoding for {suf}")
                vec = feat.groupby(field).progress_apply(
                    lambda x: bert_encoding([i for i in x[doc_field]])).to_frame()
                vec.reset_index(inplace=True)
                # vec.columns = [field, doc_field + '_vec']
                vec[doc_field + '_vec_str'] = vec[0].astype(str).apply(
                    lambda x: x[10:-3].replace(' ', '').replace('\n', ''))
                # save bert vector
                if True or self.config['save_bert_vec']:
                    # vec.to_csv(bert_vec_save_path, index=False)
                    vec[[field, doc_field + '_vec_str']].to_csv(bert_vec_save_path, index=False)
                    del vec[doc_field + '_vec_str']

            # merge bert vec to feat
            del feat[doc_field]
            feat.drop_duplicates(inplace=True)
            feat = feat[feat[field].isin(vec[field])]
            feat = pd.merge(feat, vec, on=field)
            feat.columns = [field, doc_field + '_vec']
        else:
            # if ADD_VEC, means that the model did not need docs and long_doc
            # if not ADD_VEC, means that the model need docs or long_doc.
            # load docs and long_doc
            feat[doc_field] = feat[doc_field].apply(word_map)

            # get long doc(DataFrame): user_id/item_id \t long_doc
            long_doc = feat.groupby(field).apply(lambda x: get_long_doc([i for i in x[doc_field]])).to_frame()
            long_doc.reset_index(inplace=True)

            # get docs(DataFrame): user_id/item_id \t np.array([max_sent_num, max_sent_len])
            docs = feat.groupby(field).apply(lambda x: get_docs([i for i in x[doc_field]])).to_frame()
            docs.reset_index(inplace=True)

            # merge long_doc and docs
            feat = pd.merge(docs, long_doc, on=field)

            # set head: user_id/item_id \t user_doc/item_doc \t long_user_doc/long_item_doc
            feat.columns = [field, doc_field, 'long_' + doc_field]

        return feat

    def filter_data_with_no_doc(self):
        """Remove interactions without text from both sides

        """
        self.inter_feat = self.inter_feat[self.inter_feat[self.uid_field].isin(self.user_doc[self.uid_field])]
        self.inter_feat = self.inter_feat[self.inter_feat[self.iid_field].isin(self.item_doc[self.iid_field])]

    def join(self, df):
        """Given interaction feature, join user/item doc into it.

        Args:
            df (Interaction): Interaction feature to be joint.

        Returns:
            Interaction: Interaction feature after joining operation.
        """
        df = super(PJFDataset, self).join(df)
        if self.user_doc is not None and self.uid_field in df:
            df.update(self.user_doc[df[self.uid_field]])
        if self.item_doc is not None and self.iid_field in df:
            df.update(self.item_doc[df[self.iid_field]])
        return df

    def field2feats(self, field):
        """Overload code from :class:`recbole.data.dataset`
        """
        feats = super(PJFDataset, self).field2feats(field)
        if field == self.uid_field:
            if self.user_doc is not None:
                feats.append(self.user_doc)
        elif field == self.iid_field:
            if self.item_doc is not None:
                feats.append(self.item_doc)
        return feats

    def _doc_dataframe_to_interaction(self, data):
        new_data = {}
        for k in data:
            value = data[k].values
            if k in [self.uid_field, self.iid_field]:
                new_data[k] = torch.LongTensor(value)
            else:
                new_data[k] = value
        return Interaction(new_data)

    def user_single_inter_matrix(self, form='coo', value_field=None):
        if not self.uid_field or not self.iid_field:
            raise ValueError('dataset does not exist uid/iid, thus can not converted to sparse matrix.')
        self.user_single_inter = self.inter_feat[self.inter_feat[self.direct_field] == self.geek_direct]
        return self._create_sparse_matrix(self.user_single_inter, self.uid_field, self.iid_field, form, value_field)

    def item_single_inter_matrix(self, form='coo', value_field=None):
        if not self.uid_field or not self.iid_field:
            raise ValueError('dataset does not exist uid/iid, thus can not converted to sparse matrix.')
        self.item_single_inter = self.inter_feat[self.inter_feat[self.direct_field] != self.geek_direct]
        return self._create_sparse_matrix(self.item_single_inter, self.uid_field, self.iid_field, form, value_field)

      
class SHPJFDataset(PJFDataset):
    def __init__(self, config):
        super().__init__(config)
        self._remap_qwd_id()

    def _remap_qwd_id(self):
        def word_map(sent):
            value = []
            for i, wd in enumerate(sent):
                wd = self.field2id_token['qwd_his'][wd]
                if wd not in self.wd2id.keys():
                    self.wd2id[wd] = len(self.wd2id)
                    self.id2wd.append(wd)
                value.append(self.wd2id[wd])
            return value

        self.inter_feat['qwd_his'] = self.inter_feat['qwd_his'].apply(word_map)


class PJFFFDataset(PJFDataset):
    def __init__(self, config):
        self.his_len = config['history_item_len']
        self.his_item = None
        self.his_user = None
        self.his_items_field = 'his_items'
        self.his_users_field = 'his_users'
        self.his_items_label_field = 'his_items_label'
        self.his_users_label_field = 'his_users_label'
        super().__init__(config)

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        datasets = super(PJFFFDataset, self).build()
        datasets[0]._collect_inter_his()
        datasets[0]._his_fillna()
        datasets[0]._change_his_format()

        for d in datasets:
            d.his_item = datasets[0].his_item
            d.his_user = datasets[0].his_user

        if self.config['multi_direction']:
            datasets[2].his_item = datasets[0].his_user
            datasets[2].his_user = datasets[0].his_item

            datasets[4].his_item = datasets[0].his_user
            datasets[4].his_user = datasets[0].his_item

        return datasets

    def _his_fillna(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        def fill_nan(value, fill_size):
            if isinstance(value, (list, np.ndarray, torch.Tensor)):
                return value
            else:
                return torch.zeros(fill_size).long()

        new_his_item_df = pd.DataFrame({self.uid_field: np.arange(self.user_num)})
        self.his_item = pd.merge(new_his_item_df, self.his_item, on=self.uid_field, how='left')
        self.his_item[self.his_items_field].fillna(value=0, inplace=True)
        self.his_item[self.his_items_field] = self.his_item[self.his_items_field].apply(
            fill_nan, fill_size=[self.his_len])
        self.his_item[self.his_items_label_field].fillna(value=0, inplace=True)
        self.his_item[self.his_items_label_field] = self.his_item[self.his_items_label_field].apply(
            fill_nan, fill_size=[2 * self.his_len])

        new_his_user_df = pd.DataFrame({self.iid_field: np.arange(self.item_num)})
        self.his_user = pd.merge(new_his_user_df, self.his_user, on=self.iid_field, how='left')
        self.his_user[self.his_users_field].fillna(value=0, inplace=True)
        self.his_user[self.his_users_field] = self.his_user[self.his_users_field].apply(
            fill_nan, fill_size=[self.his_len])
        self.his_user[self.his_users_label_field].fillna(value=0, inplace=True)
        self.his_user[self.his_users_label_field] = self.his_user[self.his_users_label_field].apply(
            fill_nan, fill_size=[2 * self.his_len])

    def _collect_inter_his(self):
        if self.time_field:
            self.inter_feat_all.sort_values(by=self.time_field, ascending=True)

        def get_his_ids(id_list: list):
            his_ids = np.array([]).astype(int)
            for id in id_list:
                his_ids = np.append(his_ids, id)
            if len(his_ids) > self.his_len:
                return his_ids[len(his_ids) - self.his_len:]
            return np.concatenate((his_ids, np.zeros(self.his_len - len(his_ids))), axis=0)

        self.his_user = pd.DataFrame(self.inter_feat_all).groupby(self.iid_field).apply(
            lambda x: get_his_ids([i for i in x[self.uid_field]])).to_frame()
        self.his_user.reset_index(inplace=True)
        self.his_user.columns = [self.iid_field, self.his_users_field]

        self.his_item = pd.DataFrame(self.inter_feat_all).groupby(self.uid_field).apply(
            lambda x: get_his_ids([i for i in x[self.iid_field]])).to_frame()
        self.his_item.reset_index(inplace=True)
        self.his_item.columns = [self.uid_field, self.his_items_field]

        def get_his_label(id_list: list):
            his_label = np.array([]).astype(int)
            for id in id_list:
                if id == 1:
                    his_label = np.append(his_label, [0, 1])
                else:
                    his_label = np.append(his_label, [1, 0])
            if len(his_label) > 2 * self.his_len:
                return his_label[len(his_label) - 2 * self.his_len:]
            return np.concatenate((his_label, np.zeros(2 * self.his_len - len(his_label))), axis=0).astype(int)

        if self.label_field not in self.inter_feat_all.columns:
            self.inter_feat_all[self.label_field] = 1

        self.his_item[self.his_items_label_field] = pd.DataFrame(self.inter_feat_all).groupby(self.uid_field).apply(
            lambda x: get_his_label([i for i in x[self.label_field]]))

        self.his_user[self.his_users_label_field] = pd.DataFrame(self.inter_feat_all).groupby(self.iid_field).apply(
            lambda x: get_his_label([i for i in x[self.label_field]]))

    def _change_his_format(self):
        self.his_item = self._doc_dataframe_to_interaction(self.his_item)
        self.his_user = self._doc_dataframe_to_interaction(self.his_user)

    def change_direction(self):
        """Change direction for Validation and testing.
        """
        self.uid_field, self.iid_field = self.iid_field, self.uid_field
        self.user_feat, self.item_feat = self.item_feat, self.user_feat
        self.his_item, self.his_user = self.his_user, self.his_item

    def join(self, df):
        df = super(PJFDataset, self).join(df)
        if self.his_item is not None and self.uid_field in df:
            df.update(self.his_item[df[self.uid_field]])
        if self.his_user is not None and self.iid_field in df:
            df.update(self.his_user[df[self.iid_field]])
        return df