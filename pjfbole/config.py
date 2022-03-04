# @Time   : 2022/03/02
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn


from recbole.config import Config
from pjfbole.utils import get_model

class PJFConfig(Config):
    def __init__(self, model=None, dataset=None, config_file_list=None, config_dict=None):
        super(PJFConfig, self).__init__(model, dataset, config_file_list, config_dict)


    def _get_model_and_dataset(self, model, dataset):
        if model is None:
            try:
                model = self.external_config_dict['model']
            except KeyError:
                raise KeyError(
                    'model need to be specified in at least one of the these ways: '
                    '[model variable, config file, config dict, command line] '
                )
        if not isinstance(model, str):
            final_model_class = model
            final_model = model.__name__
        else:
            final_model = model
            final_model_class = get_model(final_model)

        if dataset is None:
            try:
                final_dataset = self.external_config_dict['dataset']
            except KeyError:
                raise KeyError(
                    'dataset need to be specified in at least one of the these ways: '
                    '[dataset variable, config file, config dict, command line] '
                )
        else:
            final_dataset = dataset

        return final_model, final_model_class, final_dataset