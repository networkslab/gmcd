import pickle as pk
import os
from src.mutils import PARAM_CONFIG_FILE
from src.train_template import TrainTemplate
from src.task import TaskSyntheticModeling
from src.model.GMCD import GMCD


class TrainSyntheticModeling(TrainTemplate):
    def __init__(self,
                 runconfig,
                 batch_size,
                 checkpoint_path,
                 debug=False,
                 path_experiment="",
                 **kwargs):
        self.path_model_prefix = os.path.join(
            runconfig.dataset, "S_" + str(runconfig.S) + "_K_" + str(runconfig.K))
        super().__init__(runconfig,
                         batch_size,
                         checkpoint_path,
                         debug=debug,
                         name_prefix=path_experiment,
                         **kwargs)

    def _create_model(self, runconfig, figure_path):
        dataset_name = self.runconfig.dataset
        dataset_class = TaskSyntheticModeling.get_dataset_class(dataset_name)
        model = GMCD(run_config=runconfig,
                     dataset_class=dataset_class, figure_path=figure_path)

        return model

    def _create_task(self, runconfig, debug=False):
        task = TaskSyntheticModeling(self.model,
                                     runconfig,
                                     debug=debug,
                                     batch_size=self.batch_size)
        return task


def start_training(runconfig,
                   return_result=False,
                   store_ckpt=""):

    # Setup training
    trainModule = TrainSyntheticModeling(runconfig,
                                         batch_size=runconfig.batch_size,
                                         checkpoint_path=runconfig.checkpoint_path,
                                         path_experiment=store_ckpt)

    args_filename = os.path.join(trainModule.checkpoint_path,
                                 PARAM_CONFIG_FILE)
    with open(args_filename, "wb") as f:
        pk.dump(runconfig, f)

    # Start training

    result = trainModule.train_model(
        runconfig.max_iterations,
        loss_freq=50,
        eval_freq=runconfig.eval_freq,
        save_freq=runconfig.save_freq,
        no_model_checkpoints=runconfig.no_model_checkpoints)

    # Cleaning up the checkpoint directory afterwards if selected

    if return_result:
        return result
