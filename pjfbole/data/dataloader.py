# @Time   : 2022/3/4
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
pjfbole.data.dataloader
########################
"""

import torch

from recbole.data.dataloader.general_dataloader import TrainDataLoader, NegSampleEvalDataLoader


class DualTrainDataloader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)


class DualNegDataloader(NegSampleEvalDataLoader):
    pass
