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
from recbole.data import data_preparation
# from recbole.trainer import trainer

from pjfbole.config import PJFConfig
from pjfbole.data import create_dataset, data_preparation_for_multi_direction
from pjfbole.utils import get_model

# from recbole.utils import get_trainer
import recbole.trainer
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
    if config['multi_direction']:
        train_data, valid_g_data, valid_j_data, test_g_data, test_j_data \
            = data_preparation_for_multi_direction(config, dataset)
    else:
        train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    if config['multi_direction']:
        # model training
        best_valid_score, best_valid_result_g, best_valid_result_j = trainer.fit(
            train_data, valid_g_data, valid_j_data, saved=saved, show_progress=config['show_progress']
        )
        logger.info(set_color('best valid for geek', 'yellow') + f': {best_valid_result_g}')
        logger.info(set_color('best valid for job', 'yellow') + f': {best_valid_result_j}')

        # model evaluation
        test_result_g = trainer.evaluate(test_g_data, load_best_model=saved, show_progress=config['show_progress'])
        logger.info(set_color('test result for geek', 'yellow') + f': {test_result_g}')

        test_result_j = trainer.evaluate(test_j_data, load_best_model=saved, show_progress=config['show_progress'])
        logger.info(set_color('test result for job', 'yellow') + f': {test_result_j}')
        return {
            'best_valid_score': best_valid_score,
            'best_valid_result_g': best_valid_result_g,
            'best_valid_result_j': best_valid_result_j,
            'valid_score_bigger': config['valid_metric_bigger'],
            'test_result_g': test_result_g,
            'test_result_j': test_result_j
        }
    else:
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

