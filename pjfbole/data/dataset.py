# @Time   : 2022/03/01
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
pjfbole.data.pjf_dataset
##########################
"""

import os
import numpy as np

from recbole.data.dataset import Dataset
from recbole.utils import set_color, FeatureSource
# from pjfbole.enum_type import FeatureSource


class PJFDataset(Dataset):
    """:class:`Dataset` stores the original dataset in memory.
    It provides many useful functions for data preprocessing, such as k-core data filtering and missing value
    imputation. Features are stored as :class:`pandas.DataFrame` inside :class:`~recbole.data.dataset.dataset.Dataset`.
    General and Context-aware Models can use this class.

    By calling method :meth:`~recbole.data.dataset.dataset.Dataset.build`, it will processing dataset into
    DataLoaders, according to :class:`~recbole.config.eval_setting.EvalSetting`.

    Args:
        config (Config): Global configuration object.

    Attributes:
        dataset_name (str): Name of this dataset.

        dataset_path (str): Local file path of this dataset.

        field2type (dict): Dict mapping feature name (str) to its type (:class:`~recbole.utils.enum_type.FeatureType`).

        field2source (dict): Dict mapping feature name (str) to its source
            (:class:`~recbole.utils.enum_type.FeatureSource`).
            Specially, if feature is loaded from Arg ``additional_feat_suffix``, its source has type str,
            which is the suffix of its local file (also the suffix written in Arg ``additional_feat_suffix``).

        field2id_token (dict): Dict mapping feature name (str) to a :class:`np.ndarray`, which stores the original token
            of this feature. For example, if ``test`` is token-like feature, ``token_a`` is remapped to 1, ``token_b``
            is remapped to 2. Then ``field2id_token['test'] = ['[PAD]', 'token_a', 'token_b']``. (Note that 0 is
            always PADDING for token-like features.)

        field2token_id (dict): Dict mapping feature name (str) to a dict, which stores the token remap table
            of this feature. For example, if ``test`` is token-like feature, ``token_a`` is remapped to 1, ``token_b``
            is remapped to 2. Then ``field2token_id['test'] = {'[PAD]': 0, 'token_a': 1, 'token_b': 2}``.
            (Note that 0 is always PADDING for token-like features.)

        field2seqlen (dict): Dict mapping feature name (str) to its sequence length (int).
            For sequence features, their length can be either set in config,
            or set to the max sequence length of this feature.
            For token and float features, their length is 1.

        uid_field (str or None): The same as ``config['USER_ID_FIELD']``.

        iid_field (str or None): The same as ``config['ITEM_ID_FIELD']``.

        label_field (str or None): The same as ``config['LABEL_FIELD']``.

        time_field (str or None): The same as ``config['TIME_FIELD']``.

        inter_feat (:class:`Interaction`): Internal data structure stores the interaction features.
            It's loaded from file ``.inter``.

        user_feat (:class:`Interaction` or None): Internal data structure stores the user features.
            It's loaded from file ``.user`` if existed.

        item_feat (:class:`Interaction` or None): Internal data structure stores the item features.
            It's loaded from file ``.item`` if existed.

        feat_name_list (list): A list contains all the features' name (:class:`str`), including additional features.
    """
    def __init__(self, config):
        super().__init__(config)

    def change_direction(self):
        """Change direction for Validation and testing.
        """
        self.uid_field, self.iid_field = self.iid_field, self.uid_field
        self.user_feat, self.item_feat = self.item_feat, self.user_feat
        self.user_sents, self.item_sents = self.item_sents, self.user_sents

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

    def _load_data(self, token, dataset_path):
        """Load features.

        Firstly load interaction features, then user/item features optionally,
        finally load additional features if ``config['additional_feat_suffix']`` is set.

        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        if not os.path.exists(dataset_path):
            self._download()
        self._load_inter_feat(token, dataset_path)
        self.user_feat = self._load_user_or_item_feat(token, dataset_path, FeatureSource.USER, 'uid_field')
        self.item_feat = self._load_user_or_item_feat(token, dataset_path, FeatureSource.ITEM, 'iid_field')
        self.user_sents = self._load_user_or_item_sents(token, dataset_path, 'usents', 'uid_field')
        self.item_sents = self._load_user_or_item_sents(token, dataset_path, 'isents', 'iid_field')
        self._load_additional_feat(token, dataset_path)

    def _load_user_or_item_sents(self, token, dataset_path, suf, field_name):
        """Load user/item sents.

        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.

        Returns:
            pandas.DataFrame: Loaded sents

        Note:
            ``user_id`` and ``item_id`` has source :obj:`~recbole.utils.enum_type.FeatureSource.USER_ID` and
            :obj:`~recbole.utils.enum_type.FeatureSource.ITEM_ID`
        """
        feat_path = os.path.join(dataset_path, f'{token}.{suf}')
        field = getattr(self, field_name, None)

        if os.path.isfile(feat_path):
            feat = self._load_feat(feat_path, suf)
            self.logger.debug(f'[{suf}] feature loaded successfully from [{feat_path}].')
        else:
            feat = None
            self.logger.debug(f'[{feat_path}] not found, [{suf}] features are not loaded.')

        if feat is not None and field is None:
            raise ValueError(f'{field_name} must be exist if {suf}_feat exist.')
        if feat is not None and field not in feat:
            raise ValueError(f'{field_name} must be loaded if {suf}_feat is loaded.')
        if feat is not None:
            feat.drop_duplicates(subset=[field], keep='first', inplace=True)

        # if field in self.field2source:
        #     self.field2source[field] = FeatureSource(source.value + '_id')
        return feat
