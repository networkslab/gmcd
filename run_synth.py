from src.synthetic_run_config import SyntheticRunConfig
from src.train import start_training
from src.train_helper import print_detailed_scores_and_sampling


S = 10
dataset = 'sort'


def train(set_size, dataset):

    run_config = SyntheticRunConfig(dataset=dataset, S=set_size)
    return start_training(run_config, return_result=True)


# This will train and store the model.
detailed_scores, sample_eval = train(set_size=S,   dataset=dataset)
print_detailed_scores_and_sampling(detailed_scores, sample_eval)
