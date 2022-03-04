# @Time   : 2022/03/01
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

import copy
import importlib
import os
import pickle

from recbole.utils.argument_list import dataset_arguments
from recbole.utils import set_color

def create_dataset(config):
    dataset_module = importlib.import_module('pjfbole.data.dataset')
    if hasattr(dataset_module, config['model'] + 'Dataset'):
        dataset_class = getattr(dataset_module, config['model'] + 'Dataset')
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


def save_split_dataloaders(config, dataloaders):
    """Save split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataloaders (tuple of AbstractDataLoader): The split dataloaders.
    """
    ensure_dir(config['checkpoint_dir'])
    save_path = config['checkpoint_dir']
    saved_dataloaders_file = f'{config["dataset"]}-for-{config["model"]}-dataloader.pth'
    file_path = os.path.join(save_path, saved_dataloaders_file)
    logger = getLogger()
    logger.info(set_color('Saving split dataloaders into', 'pink') + f': [{file_path}]')
    with open(file_path, 'wb') as f:
        pickle.dump(dataloaders, f)
        

def load_split_dataloaders(config):
    """Load split dataloaders if saved dataloaders exist and
    their :attr:`config` of dataset are the same as current :attr:`config` of dataset.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        dataloaders (tuple of AbstractDataLoader or None): The split dataloaders.
    """

    default_file = os.path.join(config['checkpoint_dir'], f'{config["dataset"]}-for-{config["model"]}-dataloader.pth')
    dataloaders_save_path = config['dataloaders_save_path'] or default_file
    if not os.path.exists(dataloaders_save_path):
        return None
    with open(dataloaders_save_path, 'rb') as f:
        train_data, valid_data, test_data = pickle.load(f)
    for arg in dataset_arguments + ['seed', 'repeatable', 'eval_args']:
        if config[arg] != train_data.config[arg]:
            return None
    train_data.update_config(config)
    valid_data.update_config(config)
    test_data.update_config(config)
    logger = getLogger()
    logger.info(set_color('Load split dataloaders from', 'pink') + f': [{dataloaders_save_path}]')
    return train_data, valid_data, test_data



def data_preparation(config, dataset):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
    else:
        model_type = config['MODEL_TYPE']
        built_datasets = dataset.build()

        train_dataset, valid_dataset, test_dataset = built_datasets
        train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)

        if model_type != ModelType.KNOWLEDGE:
            train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
        else:
            kg_sampler = KGSampler(dataset, config['train_neg_sample_args']['distribution'])
            train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, kg_sampler, shuffle=True)

        valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)
        test_data = get_dataloader(config, 'evaluation')(config, test_dataset, test_sampler, shuffle=False)
        if config['save_dataloaders']:
            save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

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
    return train_data, valid_data, test_data