
import os
import numpy as np
import torch.nn as nn
import torch
import pickle
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.train_helper import check_if_best_than_saved, export_result_txt, prepare_checkpoint, print_detailed_scores_and_sampling, save_train_model_fun, store_model_dict, check_params
from src.mutils import Tracker, get_device, create_optimizer_from_args, load_model, write_dict_to_tensorboard


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
                 name_prefix=""):
        self.NUM_SAMPLES = 1000
        model_name = 'GMCD'
        path_model_prefix = os.path.join(self.path_model_prefix, model_name)
        name_prefix = os.path.join(name_prefix, path_model_prefix)
        self.batch_size = batch_size
        # Remove possible spaces. Name is used for creating default checkpoint path
        self.name_prefix = name_prefix.strip()
        self.runconfig = runconfig
        
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
                                      debug=debug)
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

    
    def train_model(self,
                    max_iterations=1e6,
                    loss_freq=50,
                    eval_freq=2000,
                    save_freq=1e5,
                    max_gradient_norm=0.25,
                    no_model_checkpoints=False):

        check_params(self.model)
        start_iter = 0
        best_save_dict = {
            "file": None,
            "metric": 1e6,
            "detailed_metrics": None,
            "test": None
        }
        best_save_iter = best_save_dict["file"]
        evaluation_dict = {}
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
        self.model.eval()
        self.task.initialize()

        print("=" * 50 + "\nStarting training...\n" + "=" * 50)

        print("Performing initial evaluation...")

        detailed_scores = self.task.eval(initial_eval=True)
        start = time.time()
        sample_eval = self.task.evaluate_sample(num_samples=self.NUM_SAMPLES)
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

        index_iter = start_iter
        keep_going = True
        self.loss_prev = None
        while keep_going:

            # Training step
            start_time = time.time()
            loss = self.task.train_step(iteration=index_iter)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()),
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

                detailed_scores = self.task.eval()
                start = time.time()
                sample_eval = self.task.evaluate_sample(
                    num_samples=self.NUM_SAMPLES, watch_z_t=True)
                end = time.time()
                time_for_sampling = (end - start)
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
                check_if_best_than_saved(last_save, detailed_scores['loss'],
                                         detailed_scores, best_save_dict,
                                         index_iter,
                                         self.get_checkpoint_filename,
                                         self.checkpoint_path,
                                         evaluation_dict, save_train_model)

            if (index_iter + 1) % save_freq == 0:
                save_train_model(index_iter + 1)
                if last_save is not None and os.path.isfile(
                        last_save) and last_save != best_save_iter:
                    print("Removing checkpoint %s..." % last_save)
                    os.remove(last_save)
                last_save = self.get_checkpoint_filename(index_iter + 1)
            index_iter += 1
            keep_going = not index_iter == int(max_iterations)
        # End training loop
        print('time to train ', np.sum(time_per_step_list))

        # Testing the trained model
        detailed_scores = self.task.test()
        print("=" * 50 + "\nTest performance: %lf" % (detailed_scores['loss']))
        detailed_scores["original_NLL"] = test_NLL
        best_save_dict["test"] = detailed_scores

        sample_eval = self.task.evaluate_sample(
            num_samples=100*self.NUM_SAMPLES)

        export_result_txt(best_save_iter, best_save_dict, self.checkpoint_path)
        writer.close()
        return detailed_scores, sample_eval

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

   
