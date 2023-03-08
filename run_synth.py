from src.synthetic_run_config import SyntheticRunConfig
from src.train import start_training


S = 6
dataset = 'sort'


def train(set_size, dataset):

    run_config = SyntheticRunConfig(dataset=dataset, S=set_size)
    start_training(run_config)


# This will train and store the model.
train(set_size=S,   dataset=dataset)

# This will evaluate the model and print the result.
