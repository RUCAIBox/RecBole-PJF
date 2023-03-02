# @Time   : 2022/3/4
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
recbole_pjf.trainer
########################
"""

from collections import OrderedDict

import torch
import numpy as np
from recbole.trainer import Trainer
from recbole.utils import calculate_valid_score


from recbole.data.interaction import Interaction
from recbole_pjf.data.dataloader import RFullSortEvalDataLoader, RNegSampleEvalDataLoader
from recbole.utils import (
    ensure_dir,
    get_local_time,
    early_stopping,
    calculate_valid_score,
    dict2str,
    EvaluatorType,
    KGDataLoaderState,
    get_tensorboard,
    set_color,
    get_gpu_usage,
    WandbLogger,
)
from tqdm import tqdm

class MultiDirectTrainer(Trainer):
    def __init__(self, config, model):
        super(MultiDirectTrainer, self).__init__(config, model)
        setattr(self.eval_collector.register,'rec.items', True)
    
    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        # Overall look at the bilateral recommendation task

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location = self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)
            
        self.model.eval()

        if isinstance(eval_data[0], RFullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            self.item_tensor = eval_data[0].dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        struct_g = self.collect_bilateral_info(eval_data[0], eval_func, show_progress, 1)
        test_result_g = self.evaluator.evaluate(struct_g)
        if not self.config["single_spec"]:
            test_result_g = self._map_reduce(test_result_g, num_sample)

        self.config.change_direction()
        if isinstance(eval_data[1], RFullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            self.item_tensor = eval_data[1].dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        struct_j = self.collect_bilateral_info(eval_data[1], eval_func, show_progress, 2)
        test_result_j = self.evaluator.evaluate(struct_j)
        if not self.config["single_spec"]:
            test_result_j = self._map_reduce(test_result_j, num_sample)
        self.config.change_direction()

        return test_result_g, test_result_j

    def _full_sort_batch_eval(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data

        inter_len = len(interaction)
        new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
        batch_size = len(new_inter)
        new_inter.update(self.item_tensor.repeat(inter_len))
        if batch_size <= self.test_batch_size:
            scores = self.model.predict(new_inter)
        else:
            scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i

    def collect_bilateral_info(self, eval_data, eval_func, show_progress, direct = 0):
        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        positive_u_list = []

        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, scores, positive_u, positive_i = eval_func(batched_data[:4])
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )

            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )

        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        return struct

    def _valid_epoch(self, valid_data, show_progress=False):
        valid_result_all = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_g_score = calculate_valid_score(valid_result_all[0], self.valid_metric)
        valid_j_score = calculate_valid_score(valid_result_all[1], self.valid_metric)
        valid_score = (valid_g_score + valid_j_score) / 2

        valid_result = OrderedDict()
        valid_result['For geek'] = valid_result_all[0]
        valid_result['\nFor job'] = valid_result_all[1]
        return valid_score, valid_result
