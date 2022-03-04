# @Time   : 2022/03/02
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
pjfbole.quick_start
########################
"""
import logging
from logging import getLogger
from recbole.utils import init_logger, init_seed, get_trainer, set_color

from pjfbole.config import PJFConfig
from pjfbole.data import create_dataset, data_preparation
from pjfbole.utils import get_model

# from recbole.utils import init_logger, get_trainer, get_model, init_seed, set_color


def run_pjfbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    # configurations initialization
    config = PJFConfig(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

