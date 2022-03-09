# @Time   : 2022/3/4
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

from collections import OrderedDict

import torch
from recbole.trainer import Trainer
from recbole.utils import calculate_valid_score


class MultiDirectTrainer(Trainer):
    def __init__(self, config, model):
        super(MultiDirectTrainer, self).__init__(config, model)

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        test_result_g = super(MultiDirectTrainer, self).evaluate(eval_data[0], load_best_model=load_best_model,
                                                                 model_file=model_file, show_progress=show_progress)

        self.config.change_direction()
        test_result_j = super(MultiDirectTrainer, self).evaluate(eval_data[1], load_best_model=load_best_model,
                                                                 model_file=model_file, show_progress=show_progress)
        self.config.change_direction()
        return test_result_g, test_result_j

    def _valid_epoch(self, valid_data, show_progress=False):
        valid_result_all = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_g_score = calculate_valid_score(valid_result_all[0], self.valid_metric)
        valid_j_score = calculate_valid_score(valid_result_all[1], self.valid_metric)
        valid_score = (valid_g_score + valid_j_score) / 2

        valid_result = OrderedDict()
        valid_result['For geek'] = valid_result_all[0]
        valid_result['\nFor job'] = valid_result_all[1]
        return valid_score, valid_result
