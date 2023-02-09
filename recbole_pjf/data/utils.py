# @Time   : 2022/03/01
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
recbole_pjf.data.utils
########################
"""

import importlib
import os
import pickle
from logging import getLogger

from recbole.sampler import Sampler, RepeatableSampler
from recbole.utils.argument_list import dataset_arguments
from recbole.utils import set_color
from recbole.data.utils import data_preparation as recbole_data_preparation
from recbole.data.dataloader import *

from recbole_pjf.data.dataloader import *


def create_dataset(config):
    """Create dataset according to :attr:`config['model']`.
    If :attr:`config['dataset_save_path']` file exists and
    its :attr:`config` of dataset is equal to current :attr:`config` of dataset.
    It will return the saved dataset in :attr:`config['dataset_save_path']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    dataset_module = importlib.import_module('recbole_pjf.data.dataset')
    recbole_dataset_module = importlib.import_module('recbole.data.dataset')
    if hasattr(dataset_module, config['model'] + 'Dataset'):
        dataset_class = getattr(dataset_module, config['model'] + 'Dataset')
    elif hasattr(recbole_dataset_module, config['model'] + 'Dataset'):
        dataset_class = getattr(recbole_dataset_module, config['model'] + 'Dataset')
    else:
        dataset_class = getattr(dataset_module, 'PJFDataset')

    default_file = os.path.join(config['checkpoint_dir'], f'{config["dataset"]}-{dataset_class.__name__}.pth')
    file = config['dataset_save_path'] or default_file
    if os.path.exists(file):
        with open(file, 'rb') as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ['seed', 'repeatable']:
            if config[arg] != dataset.config[arg]:
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            logger = getLogger()
            logger.info(set_color('Load filtered dataset from', 'pink') + f': [{file}]')
            return dataset

    dataset = dataset_class(config)
    if config['save_dataset']:
        dataset.save()
    return dataset


def data_preparation(config, dataset):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.
    If the :attr:`config['biliteral']` is set, the dataset is divided into
    train, valid for geek, valid for job, test for geek and test for job.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - (valid_g_data, valid_j_data) (tuple of AbstractDataLoader): The dataloader for validation for geek and job.
            - (test_g_data, test_j_data) (tuple of AbstractDataLoader): The dataloader for testing for geek and job.
    """
    if not config['biliteral']:
        return recbole_data_preparation(config, dataset)

    built_datasets = dataset.build()

    train_dataset, valid_g_dataset, valid_j_dataset, test_g_dataset, test_j_dataset = built_datasets
    train_sampler, valid_g_sampler, valid_j_sampler, test_g_sampler, test_j_sampler \
        = create_samplers_for_multi_direction(config, dataset, built_datasets)

    train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
    valid_g_data = get_dataloader(config, 'evaluation')(config, valid_g_dataset, valid_g_sampler, shuffle=False)
    valid_j_data = get_dataloader(config, 'evaluation')(config, valid_j_dataset, valid_j_sampler, shuffle=False)
    test_g_data = get_dataloader(config, 'evaluation')(config, test_g_dataset, test_g_sampler, shuffle=False)
    test_j_data = get_dataloader(config, 'evaluation')(config, test_j_dataset, test_j_sampler, shuffle=False)

    logger = getLogger()
    logger.info(
        set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["train_batch_size"]}]', 'yellow') + set_color(' negative sampling', 'cyan') + ': ' +
        set_color(f'[{config["neg_sampling"]}]', 'yellow')
    )
    logger.info(
        set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["eval_batch_size"]}]', 'yellow') + set_color(' eval_args', 'cyan') + ': ' +
        set_color(f'[{config["eval_args"]}]', 'yellow')
    )
    return train_data, (valid_g_data, valid_j_data), (test_g_data, test_j_data)


def create_samplers_for_multi_direction(config, dataset, built_datasets):
    """Create sampler for training, validation for geek and job, and testing for geek and job.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation for geek and job, and testing for geek and job.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_g_sampler (AbstractSampler): The sampler for validation for geek.
            - valid_j_sampler (AbstractSampler): The sampler for validation for job.
            - test_g_sampler (AbstractSampler): The sampler for testing for geek.
            - test_j_sampler (AbstractSampler): The sampler for testing for job.
    """
    phases = ['train', 'valid_g', 'valid_j', 'test_g', 'test_j']
    train_neg_sample_args = config['train_neg_sample_args']
    eval_neg_sample_args = config['eval_neg_sample_args']
    g_sampler, j_sampler = None, None
    valid_g_sampler, valid_j_sampler, test_g_sampler, test_j_sampler = None, None, None, None

    if train_neg_sample_args['distribution'] != 'none':
        if not config['repeatable']:
            sampler = Sampler(phases, built_datasets, train_neg_sample_args['distribution'])
        else:
            sampler = RepeatableSampler(phases, dataset, train_neg_sample_args['distribution'])
        train_sampler = sampler.set_phase('train')

    if eval_neg_sample_args['distribution'] != 'none':
        if g_sampler is None or j_sampler is None:
            if not config['repeatable']:
                g_sampler = Sampler(phases, built_datasets, eval_neg_sample_args['distribution'])
                built_datasets[0].change_direction()
                j_sampler = Sampler(phases, built_datasets, eval_neg_sample_args['distribution'])
            else:
                g_sampler = RepeatableSampler(phases, dataset, eval_neg_sample_args['distribution'])
                built_datasets[0].change_direction()
                j_sampler = RepeatableSampler(phases, dataset, eval_neg_sample_args['distribution'])
        else:
            g_sampler.set_distribution(eval_neg_sample_args['distribution'])
            built_datasets[0].change_direction()
            j_sampler.set_distribution(eval_neg_sample_args['distribution'])
            
        valid_g_sampler = g_sampler.set_phase('valid_g')
        valid_j_sampler = j_sampler.set_phase('valid_j')
        test_g_sampler = g_sampler.set_phase('test_g')
        test_j_sampler = j_sampler.set_phase('test_j')

        built_datasets[0].change_direction()

    return train_sampler, valid_g_sampler, valid_j_sampler, test_g_sampler, test_j_sampler


def get_dataloader(config, phase):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    model_type = config['MODEL_TYPE']
    model_name = config['model']
    if phase == 'train':
        try:
            return getattr(importlib.import_module('recbole_pjf.data.dataloader'), model_name + 'TrainDataloader')
        except AttributeError:
            if model_type != ModelType.KNOWLEDGE:
                return PJFTrainDataLoader
            else:
                return KnowledgeBasedDataLoader
    else:
        eval_strategy = config['eval_args']['mode']
        if eval_strategy == "full":
            return RFullSortEvalDataLoader
        else:
            return RNegSampleEvalDataLoader
            
