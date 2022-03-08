# @Time   : 2022/03/02
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
pjfbole.utils
########################
"""

import importlib


def get_model(model_name):
    model_file_name = model_name.lower()
    model_module = None
    module_path = '.'.join(['pjfbole.model', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))
    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer(model_type, model_name, multi_direction=False):
    try:
        return getattr(importlib.import_module('pjfbole.trainer'), model_name + 'Trainer')
    except AttributeError:
        if multi_direction:
            return getattr(importlib.import_module('pjfbole.trainer'), 'MultiDirectTrainer')
        else:
            return getattr(importlib.import_module('recbole.trainer'), 'Trainer')