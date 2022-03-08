# @Time   : 2022/03/01
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
recbole.data.pjf_dataset
##########################
"""

import numpy as np

from recbole.data.dataset import Dataset
from recbole.utils import set_color


class PJFDataset(Dataset):
    def __init__(self, config):
        super().__init__(config)
        self.direct = 0
        # self._set_dual_side_data()

    def _remap_ID_all(self):
        """Remap all token-like fields.
        """
        for alias in self.alias.values():
            remap_list = self._get_remap_list(alias)
            self._remap(remap_list)

        for field in self._rest_fields:
            remap_list = self._get_remap_list(np.array([field]))
            self._remap(remap_list)

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        self._change_feat_format()

        if self.benchmark_filename_list is not None:
            cumsum = list(np.cumsum(self.file_size_list))
            datasets = [self.copy(self.inter_feat[start:end]) for start, end in zip([0] + cumsum[:-1], cumsum)]
        else:
            # ordering
            ordering_args = self.config['eval_args']['order']
            if ordering_args == 'RO':
                self.shuffle()
            elif ordering_args == 'TO':
                self.sort(by=self.time_field)
            else:
                raise NotImplementedError(f'The ordering_method [{ordering_args}] has not been implemented.')

            # splitting & grouping
            split_args = self.config['eval_args']['split']
            if split_args is None:
                raise ValueError('The split_args in eval_args should not be None.')
            if not isinstance(split_args, dict):
                raise ValueError(f'The split_args [{split_args}] should be a dict.')

            split_mode = list(split_args.keys())[0]
            assert len(split_args.keys()) == 1
            group_by = self.config['eval_args']['group_by']
            if split_mode == 'RS':
                if not isinstance(split_args['RS'], list):
                    raise ValueError(f'The value of "RS" [{split_args}] should be a list.')
                if group_by is None or group_by.lower() == 'none':
                    datasets = self.split_by_ratio(split_args['RS'], group_by=None)
                elif group_by == 'user':
                    datasets = self.split_by_ratio(split_args['RS'], group_by=self.uid_field)
                else:
                    raise NotImplementedError(f'The grouping method [{group_by}] has not been implemented.')
            elif split_mode == 'LS':
                datasets = self.leave_one_out(group_by=self.uid_field, leave_one_mode=split_args['LS'])
            else:
                raise NotImplementedError(f'The splitting_method [{split_mode}] has not been implemented.')

        if self.config['multi_direction']:
            d = self.config['DIRECT_FIELD']
            geek_direct = datasets[0].field2token_id[d]['0']
            valid_g = self.copy(datasets[1].inter_feat[datasets[1].inter_feat[d] == geek_direct])

            valid_j = self.copy(datasets[1].inter_feat[datasets[1].inter_feat[d] != geek_direct])
            valid_j.uid_field, valid_j.iid_field = valid_j.iid_field, valid_j.uid_field

            test_g = self.copy(datasets[2].inter_feat[datasets[2].inter_feat[d] == geek_direct])

            test_j = self.copy(datasets[2].inter_feat[datasets[2].inter_feat[d] != geek_direct])
            test_j.uid_field, test_j.iid_field = test_j.iid_field, test_j.uid_field
            return [datasets[0], valid_g, valid_j, test_g, test_j]
        return datasets


    # def _get_field_from_config(self):
    #     """Initialization common field names.
    #     """
    #     self.uid_field = self.config['USER_ID_FIELD']
    #     self.iid_field = self.config['ITEM_ID_FIELD']
    #     self.label_field = self.config['LABEL_FIELD']
    #     self.time_field = self.config['TIME_FIELD']
    #     # self.direct_field = self.config['DIRECT_FIELD']
    #
    #     if (self.uid_field is None) ^ (self.iid_field is None):
    #         raise ValueError(
    #             'USER_ID_FIELD and ITEM_ID_FIELD need to be set at the same time or not set at the same time.'
    #         )
    #
    #     self.logger.debug(set_color('uid_field', 'blue') + f': {self.uid_field}')
    #     self.logger.debug(set_color('iid_field', 'blue') + f': {self.iid_field}')

    # def _set_dual_side_data(self):
    #     if self.config['multi_direction']:
    #         self.geek_inter = self.inter_feat[self.inter_feat['direct'] == '0']
    #         self.job_inter = self.inter_feat[self.inter_feat['direct'] == '1']
    #
    # def set_direction(self, direct):
    #     self.direct = direct
    #
    # def __getitem__(self, index, join=True):
    #     if self.direct == 0:
    #         df = self.geek_inter[index]
    #     else:
    #         df = self.job_inter[index]
    #     return self.join(df) if join else df
