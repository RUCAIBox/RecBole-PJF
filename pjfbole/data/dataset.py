# @Time   : 2022/03/01
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
pjfbole.data.pjf_dataset
##########################
"""

import numpy as np

from recbole.data.dataset import Dataset
from recbole.utils import set_color


class PJFDataset(Dataset):
    """ Configurator module that load the defined parameters.

    Configurator module will first load the default parameters from the fixed properties in RecBole and then
    load parameters from the external input.

    External input supports three kind of forms: config file, command line and parameter dictionaries.

    - config file: It's a file that record the parameters to be modified or added. It should be in ``yaml`` format,
      e.g. a config file is 'example.yaml', the content is:

        learning_rate: 0.001

        train_batch_size: 2048

    - command line: It should be in the format as '---learning_rate=0.001'

    - parameter dictionaries: It should be a dict, where the key is parameter name and the value is parameter value,
      e.g. config_dict = {'learning_rate': 0.001}

    Configuration module allows the above three kind of external input format to be used together,
    the priority order is as following:

    command line > parameter dictionaries > config file

    e.g. If we set learning_rate=0.01 in config file, learning_rate=0.02 in command line,
    learning_rate=0.03 in parameter dictionaries.

    Finally the learning_rate is equal to 0.02.
    """
    def __init__(self, config):
        super().__init__(config)

    def _remap_ID_all(self):
        """Remap all token-like fields.
        """
        for alias in self.alias.values():
            remap_list = self._get_remap_list(alias)
            self._remap(remap_list)

        for field in self._rest_fields:
            remap_list = self._get_remap_list(np.array([field]))
            self._remap(remap_list)

    def change_direction(self):
        """Change direction for Validation and testing.
        """
        self.uid_field, self.iid_field = self.iid_field, self.uid_field

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
            valid_j.change_direction()

            test_g = self.copy(datasets[2].inter_feat[datasets[2].inter_feat[d] == geek_direct])

            test_j = self.copy(datasets[2].inter_feat[datasets[2].inter_feat[d] != geek_direct])
            test_j.change_direction()
            return [datasets[0], valid_g, valid_j, test_g, test_j]
        return datasets

