
import os
import numpy as np
import torch.nn as nn
import torch
import pickle
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.train_helper import check_if_best_than_saved, export_result_txt, model_param_to_name, prepare_checkpoint, print_detailed_scores_and_sampling, save_train_model_fun, store_model_dict, check_params
from src.mutils import Tracker, get_device,  get_param_val, create_optimizer_from_args, load_model, write_dict_to_tensorboard


class TrainTemplate:
    """
    Template class to handle the training loop.
    Each experiment contains a experiment-specific training class inherting from this template class.
    """

    def __init__(self,
                 runconfig,
                 batch_size,
                 checkpoint_path,
                 debug=False,
                 name_prefix="",
                 multi_gpu=False,
                 silence=False):
        self.NUM_SAMPLES = 1000
        model_name = model_param_to_name(runconfig)
        path_model_prefix = os.path.join(self.path_model_prefix, model_name)
        name_prefix = os.path.join(name_prefix, path_model_prefix)
        self.batch_size = batch_size
        # Remove possible spaces. Name is used for creating default checkpoint path
        self.name_prefix = name_prefix.strip()
        self.runconfig = runconfig
        self.silence = silence
        self.scale_loss = self.runconfig.scale_loss

        self.checkpoint_path, self.figure_path = prepare_checkpoint(
            checkpoint_path, self.name_prefix)
        # store model cinfo
        store_model_dict(self.figure_path, runconfig)
        runconfig.checkpoint_path = self.checkpoint_path
        # Load model
        self.model = self._create_model(runconfig, self.figure_path)
        self.model = self.model.to(get_device())

        # Load task
        self.task = self._create_task(runconfig,
                                      debug=debug,
                                      silence=self.silence)
        # Load optimizer and checkpoints
        self._create_optimizer(runconfig)

    def _create_optimizer(self, optimizer_params):
        parameters_to_optimize = self.model.parameters()
        self.optimizer = create_optimizer_from_args(parameters_to_optimize,
                                                    optimizer_params)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            optimizer_params.lr_decay_step,
            gamma=optimizer_params.lr_decay_factor)
        self.lr_minimum = optimizer_params.lr_minimum

    def nll_evaluation(self, silence=False, checkpoint_model=None, return_result=False):
        # Function for evaluation/testing of a model

        # Load the "best" model by first loading the most recent one and determining the "best" model
        checkpoint_dict = self.load_recent_model()
        best_save_dict = get_param_val(checkpoint_dict,
                                       "best_save_dict", {
                                           "file": None,
                                           "metric": -1,
                                           "detailed_metrics": dict()
                                       },
                                       warning_if_default=True)
        best_save_iter = best_save_dict["file"]
        if not os.path.isfile(best_save_iter):
            splits = best_save_iter.split("/")
            checkpoint_index = splits.index("checkpoints")
            best_save_iter = "/".join(splits[checkpoint_index:])
        if not os.path.isfile(best_save_iter):
            print(
                "[!] WARNING: Tried to load best model \"%s\", but file does not exist"
                % (best_save_iter))
        else:
            load_model(best_save_iter, model=self.model)
        if not silence:
            # Print saved information of performance on validation set
            print("\n" + "-" * 100 + "\n")
            print("Best evaluation iteration", best_save_iter)
            print("Best evaluation metric", best_save_dict["metric"])
            print("Detailed metrics")
            for metric_name, metric_val in best_save_dict[
                    "detailed_metrics"].items():
                print("-> %s: %s" % (metric_name, str(metric_val)))
            print("\n" + "-" * 100 + "\n")

        # Test model
        self.task.checkpoint_path = self.checkpoint_path
        eval_metric, detailed_metrics = self.task.test()

        return detailed_metrics

    def complete_evaluation(self, num_trial=4, silence=False):
        print('Starting the eval....')
        check_params(self.model, silence=False)
        NUM_SAMPLES = 10000

        index = 0
        recompute_stat = False
        print('recomputing stat :', recompute_stat)
        detailed_metrics_test = {}
        try:
            detailed_metrics_test = self.nll_evaluation(silence=silence)
        except:
            print('couldnt get a best iter')
            self.load_recent_model()

        empty_sample_eval = self.task.evaluate_sample(
            num_samples=1, watch_z_t=False, return_empty=True)
        list_sample_eval = []
        if empty_sample_eval.is_stored(self.figure_path, index):
            sample_eval = empty_sample_eval.get(self.figure_path, index)
            # recompute the stats
            if recompute_stat:
                sample_eval = self.task.get_sample_eval(sample_eval.samples)
        else:
            start = time.time()
            sample_eval = self.task.evaluate_sample(num_samples=NUM_SAMPLES,
                                                    watch_z_t=True)

            sample_eval.store(self.figure_path, index)
            end = time.time()
            time_for_sampling = (end - start)
            print('time_for_sampling', time_for_sampling)

        list_sample_eval.append(sample_eval)
        if silence:
            iter_trial = range(num_trial-1)
        else:
            iter_trial = tqdm(range(num_trial-1))
        for _ in iter_trial:
            index += 1
            if empty_sample_eval.is_stored(self.figure_path, index):
                sample_eval = empty_sample_eval.get(self.figure_path, index)
                if recompute_stat:
                    sample_eval = self.task.get_sample_eval(
                        sample_eval.samples)
            else:

                sample_eval = self.task.evaluate_sample(num_samples=NUM_SAMPLES,
                                                        watch_z_t=False)

            sample_eval.store(self.figure_path, index)

            list_sample_eval.append(sample_eval)

        index = 100
        if empty_sample_eval.is_stored(self.figure_path, index):
            big_sample_eval = empty_sample_eval.get(self.figure_path, index)
            if recompute_stat:
                sample_eval = self.task.get_sample_eval(sample_eval.samples)

        else:
            # at the end , concat everything and compute stat on the big samples
            list_samples = [
                sample_eval.samples for sample_eval in list_sample_eval]
            big_samples = np.concatenate(list_samples, axis=0)

            big_sample_eval = self.task.get_sample_eval(big_samples)
        big_sample_eval.store(self.figure_path, index)

        return list_sample_eval, big_sample_eval, detailed_metrics_test

    def train_model(self,
                    max_iterations=1e6,
                    loss_freq=50,
                    eval_freq=2000,
                    save_freq=1e5,
                    max_gradient_norm=0.25,
                    no_model_checkpoints=False,
                    max_train_time=None):

        parameters_to_optimize = list(self.model.parameters())

        check_params(self.model, silence=self.silence)
        # Setup dictionary to save evaluation details in
        checkpoint_dict = self.load_recent_model()
        # Iteration to start from
        start_iter = get_param_val(checkpoint_dict,
                                   "iteration",
                                   0,
                                   warning_if_default=False)
        evaluation_dict = get_param_val(
            checkpoint_dict,
            "evaluation_dict",
            dict(),
            warning_if_default=False
        )  # Dictionary containing validation performances over time
        best_save_dict = get_param_val(checkpoint_dict,
                                       "best_save_dict", {
                                           "file": None,
                                           "metric": 1e6,
                                           "detailed_metrics": None,
                                           "test": None
                                       },
                                       warning_if_default=False)
        best_save_iter = best_save_dict["file"]
        last_save = None if start_iter == 0 else self.get_checkpoint_filename(
            start_iter)
        if last_save is not None and not os.path.isfile(last_save):
            print(
                "[!] WARNING: Could not find last checkpoint file specified as "
                + last_save)
            last_save = None
        test_NLL = None  # Possible test performance determined in the end of the training

        def save_train_model(index_iter):
            return save_train_model_fun(no_model_checkpoints,
                                        best_save_dict,
                                        evaluation_dict,
                                        index_iter,
                                        self.save_model,
                                        only_weights=True)

        # Initialize tensorboard writer
        writer = SummaryWriter(self.checkpoint_path)

        # "Trackers" are moving averages. We use them to log the loss and time needed per training iteration
        time_per_step = Tracker()
        time_per_step_list = []
        train_losses = Tracker()
        # Try-catch if user terminates
        try:

            self.model.eval()
            self.task.initialize()
            if not self.silence:
                print("=" * 50 + "\nStarting training...\n" + "=" * 50)

                print("Performing initial evaluation...")

            eval_loss, detailed_scores = self.task.eval(initial_eval=True)
            start = time.time()

            # self.task.check_dead_zone('start_training_')

            sample_eval = self.task.evaluate_sample(num_samples=self.NUM_SAMPLES,
                                                    watch_z_t=False)
            end = time.time()
            time_for_sampling = (end - start)

            print_detailed_scores_and_sampling(detailed_scores,
                                               sample_eval)
            print('time for sampling ', self.NUM_SAMPLES, ' samples : ',
                  "{:.2f}".format((time_for_sampling)), ' sec')

            self.model.train()
            detailed_scores_to_tensorboard = {}
            detailed_scores_to_tensorboard.update(
                sample_eval.get_printable_metrics_dict())
            detailed_scores_to_tensorboard.update(detailed_scores)
            write_dict_to_tensorboard(writer,
                                      detailed_scores_to_tensorboard,
                                      base_name="eval",
                                      iteration=start_iter)

            def get_keep_going(time_per_step_list, index_iter):
                if max_train_time is not None:
                    time_to_train = np.sum(time_per_step_list)
                    return time_to_train < max_train_time
                else:
                    return not index_iter == int(max_iterations)

            index_iter = start_iter
            keep_going = True
            self.loss_prev = None
            while keep_going:

                # Training step
                start_time = time.time()
                loss = self.task.train_step(iteration=index_iter)

                self.optimizer.zero_grad()
                loss.backward()
               
                torch.nn.utils.clip_grad_norm_(parameters_to_optimize,
                                               max_gradient_norm)
                self.optimizer.step()
                if self.optimizer.param_groups[0]['lr'] > self.lr_minimum:
                    self.lr_scheduler.step()
                end_time = time.time()

                time_per_step.add(end_time - start_time)
                time_per_step_list.append(end_time - start_time)
                train_losses.add(loss.item())

                if (index_iter + 1) % loss_freq == 0:

                    loss_avg = train_losses.get_mean(reset=True)
                    self.loss_prev = loss_avg
                    train_time_avg = time_per_step.get_mean(reset=True)
                    if not self.silence:
                        print(
                            "Training iteration %i|%i (%4.2fs). Loss: %6.5f." %
                            (index_iter + 1, max_iterations, train_time_avg,
                             loss_avg))
                    writer.add_scalar("train/loss", loss_avg, index_iter + 1)
                    writer.add_scalar("train/learning_rate",
                                      self.optimizer.param_groups[0]['lr'],
                                      index_iter + 1)
                    writer.add_scalar("train/training_time", train_time_avg,
                                      index_iter + 1)

                    self.task.add_summary(writer,
                                          index_iter + 1,
                                          checkpoint_path=self.checkpoint_path)

                # Performing evaluation every "eval_freq" steps
                if (index_iter + 1) % eval_freq == 0:
                    self.model.eval()

                    eval_loss, detailed_scores = self.task.eval()
                    start = time.time()
                    sample_eval = self.task.evaluate_sample(
                        num_samples=self.NUM_SAMPLES, watch_z_t=True)
                    end = time.time()
                    time_for_sampling = (end - start)
                    if not self.silence:
                        print_detailed_scores_and_sampling(
                            detailed_scores, sample_eval)

                        print('time for sampling ', self.NUM_SAMPLES, ' samples : ',
                              "{:.2f}".format((time_for_sampling)), ' sec')
                    if 'overfit_detected' in sample_eval.all_dict:
                        if sample_eval.all_dict['overfit_detected']:
                            print('model overfitting...')
                            keep_going = False
                    self.model.train()
                    detailed_scores_to_tensorboard = {}
                    detailed_scores_to_tensorboard.update(
                        sample_eval.get_printable_metrics_dict())
                    detailed_scores_to_tensorboard.update(detailed_scores)
                    write_dict_to_tensorboard(writer,
                                              detailed_scores_to_tensorboard,
                                              base_name="eval",
                                              iteration=index_iter + 1)

                    # If model performed better on validation than any other iteration so far => save it and eventually replace old model
                    check_if_best_than_saved(self.silence, last_save, eval_loss,
                                             detailed_scores, best_save_dict,
                                             index_iter,
                                             self.get_checkpoint_filename,
                                             self.checkpoint_path,
                                             self.task.export_best_results,
                                             evaluation_dict, save_train_model)

                # Independent of evaluation, the model is saved every "save_freq" steps. This prevents loss of information if model does not improve for a while
                if (index_iter + 1) % save_freq == 0 and not os.path.isfile(
                        self.get_checkpoint_filename(index_iter + 1)):
                    save_train_model(index_iter + 1)
                    if last_save is not None and os.path.isfile(
                            last_save) and last_save != best_save_iter:
                        if not self.silence:
                            print("Removing checkpoint %s..." % last_save)
                        os.remove(last_save)
                    last_save = self.get_checkpoint_filename(index_iter + 1)
                index_iter += 1
                keep_going = get_keep_going(time_per_step_list, index_iter)
            # End training loop
            if not self.silence:
                print('time to train ', np.sum(time_per_step_list))

            # Testing the trained model
            test_NLL, detailed_scores = self.task.test()
            if not self.silence:
                print("=" * 50 + "\nTest performance: %lf" % (test_NLL))
            detailed_scores["original_NLL"] = test_NLL
            best_save_dict["test"] = detailed_scores
            self.task.finalize_summary(writer, max_iterations,
                                       self.checkpoint_path)

            sample_eval = self.task.evaluate_sample(num_samples=10 *
                                                    self.NUM_SAMPLES,
                                                    watch_z_t=True)

            detailed_scores.update(sample_eval.get_all_metrics_dict())

        # If user terminates training early, replace last model saved per "save_freq" steps by current one
        except KeyboardInterrupt:
            if index_iter > 0:
                print(
                    "User keyboard interrupt detected. Saving model at step %i..."
                    % (index_iter))
                save_train_model(index_iter + 1)
            else:
                print(
                    "User keyboard interrupt detected before starting to train."
                )
            if last_save is not None and os.path.isfile(last_save) and not any(
                    [val == last_save for _, val in best_save_dict.items()]):
                os.remove(last_save)

        export_result_txt(best_save_iter, best_save_dict, self.checkpoint_path)
        writer.close()
        return detailed_scores

    def get_checkpoint_filename(self, iteration):
        checkpoint_file = os.path.join(
            self.checkpoint_path,
            'checkpoint_' + str(iteration).zfill(7) + ".tar")
        return checkpoint_file

    def save_model(self,
                   iteration,
                   add_param_dict,
                   save_embeddings=False,
                   save_optimizer=True):
        checkpoint_file = self.get_checkpoint_filename(iteration)
        if isinstance(self.model, nn.DataParallel):
            model_dict = self.model.module.state_dict()
        else:
            model_dict = self.model.state_dict()

        checkpoint_dict = {'model_state_dict': model_dict}
        if save_optimizer:
            checkpoint_dict[
                'optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint_dict[
                'scheduler_state_dict'] = self.lr_scheduler.state_dict()
        checkpoint_dict.update(add_param_dict)
        torch.save(checkpoint_dict, checkpoint_file)

    def load_recent_model(self):
        checkpoint_dict = load_model(self.checkpoint_path,
                                     model=self.model,
                                     optimizer=self.optimizer,
                                     lr_scheduler=self.lr_scheduler)
        return checkpoint_dict
