# @Time   : 2022/3/4
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

import os
from logging import getLogger
from time import time

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from recbole.trainer import Trainer
from recbole.utils import set_color, dict2str, early_stopping, calculate_valid_score


class MultiDirectTrainer(Trainer):
    def __init__(self, config, model):
        super(MultiDirectTrainer, self).__init__(config, model)
        self.best_valid_g_result = None
        self.best_valid_j_result = None

    def fit(self, train_data, valid_g_data=None, valid_j_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config['train_neg_sample_args'].get('dynamic', 'none') != 'none':
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step':epoch_idx}, head='train')

            # eval
            if self.eval_step <= 0 or not valid_g_data or not valid_j_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_g_score, valid_g_result = self._valid_epoch(valid_g_data, show_progress=show_progress)
                valid_j_score, valid_j_result = self._valid_epoch(valid_j_data, show_progress=show_progress, reverse=True) # for evaluate in job direction
                valid_score = (valid_g_score + valid_j_score) / 2

                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' \
                                      + 'for geek:' + dict2str(valid_g_result) + '\n' \
                                      + 'for job:' + dict2str(valid_j_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)

                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                self.wandblogger.log_metrics({**valid_g_result, 'valid_step': valid_step}, head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_g_result = valid_g_result
                    self.best_valid_j_result = valid_j_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_g_result, self.best_valid_j_result

    def _valid_epoch(self, valid_data, show_progress=False, reverse=False):
        if reverse:
            self.config['ITEM_ID_FIELD'], self.config['USER_ID_FIELD'] \
                = self.config['USER_ID_FIELD'], self.config['ITEM_ID_FIELD']
        valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        if reverse:
            self.config['ITEM_ID_FIELD'], self.config['USER_ID_FIELD'] \
                = self.config['USER_ID_FIELD'], self.config['ITEM_ID_FIELD']
        return valid_score, valid_result
