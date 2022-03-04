from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import ModelType, InputType

class PJFNN(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(PJFNN, self).__init__(config, dataset)