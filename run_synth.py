import argparse
from src.experiment.synthetic_run_config import SyntheticRunConfig
from src.experiment.synthetic_train import start_training
from src.train_helper import print_detailed_scores_and_sampling


def train(set_size):

    run_config = SyntheticRunConfig(dataset='sort', S=set_size)
    return start_training(run_config, return_result=True)


if __name__ == '__main__':
    # This will train and store the model.

    parser = argparse.ArgumentParser()
    parser.add_argument('--S', type=int, default=6,
                        help='synthetic experiment dimension')
    args = parser.parse_args()
    detailed_scores, sample_eval = train(set_size=args.S)
    print_detailed_scores_and_sampling(detailed_scores, sample_eval)
