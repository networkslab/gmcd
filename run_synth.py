from src.synthetic_run_config import SyntheticRunConfig
from src.train import start_training


S = 6
dataset = 'sort'


def train(set_size, dataset):

    run_config = SyntheticRunConfig(dataset=dataset, S=set_size)
    return start_training(run_config, return_result=True)


# This will train and store the model.
result = train(set_size=S,   dataset=dataset)

print(result)
