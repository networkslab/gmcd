
from tqdm import tqdm
from src.mutils import get_device
from src.metrics import *
import pickle as pk
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from statistics import mean
import time
import os
from src.sample_output import SamplesEvaluation
import math


class TaskTemplate:

    def __init__(self, model, run_config, name, load_data=True, debug=False, batch_size=64, drop_last=False, num_workers=None):
        # Saving parameters
        self.name = name
        self.model = model
        self.run_config = run_config
        self.batch_size = batch_size
        self.train_batch_size = batch_size
        self.debug = debug
        self.max_samples = 1000
        # Initializing dataset parameters
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_epoch = 0
        # Load data if specified, and create data loaders
        if load_data:
            self._load_datasets()
            self._initialize_data_loaders(
                drop_last=drop_last, num_workers=num_workers)
        else:
            self.train_data_loader = None
            self.train_data_loader_iter = None
            self.val_data_loader = None
            self.test_data_loader = None

        # Create a dictionary to store summary metrics in
        self.summary_dict = {}

        # Placeholders for visualization
        self.gen_batch = None
        self.class_colors = None

        # Put model on correct device
        self.model.to(get_device())
        self._precompute_stats_for_metrics()

    def _precompute_stats_for_metrics(self):
        # if the metrics already has been precomputed, load it.
        # Pairwise Covariance
        data_path = self.test_dataset.data_path
        self.higher_order_dict_n_abs = {}
        for n in range(2, min(10, self.model.S)):
            file = str(n)+'_list_highcov_test_abs.pk'
            filepath = os.path.join(data_path, file)

            if not os.path.exists(filepath):
                higher_order_dict_abs = compute_higher_order_stats(
                    self.test_dataset.np_data, num_patterns=1000, n=n, absolute_version=True)

                with open(filepath, 'wb') as f:
                    pk.dump(higher_order_dict_abs, f)
            else:
                with open(filepath, 'rb') as f:
                    higher_order_dict_abs = pk.load(f)
            self.higher_order_dict_n_abs[n] = higher_order_dict_abs
        filepath = os.path.join(data_path, 'training_dict.pk')
        if not os.path.exists(filepath):
            training_dict = self.data_to_dict(
                self.train_dataset.np_data)  # todo load and store
            with open(filepath, 'wb') as f:
                pk.dump(training_dict, f)
        else:
            with open(filepath, 'rb') as f:
                training_dict = pk.load(f)
        filepath = os.path.join(data_path, 'test_dict.pk')
        if not os.path.exists(filepath):
            test_dict = self.data_to_dict(
                self.test_dataset.np_data)  # todo load and store
            with open(filepath, 'wb') as f:
                pk.dump(test_dict, f)
        else:
            with open(filepath, 'rb') as f:
                test_dict = pk.load(f)
        self.training_dict = training_dict
        self.test_dict = test_dict

        self.frac_val_seen = get_frac_overlap(
            self.training_dict, self.test_dict)


    def data_to_dict(self, np_data):
        dict_data = {}
        for i in tqdm(range(np_data.shape[0])):
            x = list(np_data[i, :])
            str_x = '-'.join(str(s) for s in x)
            if str_x in dict_data:
                dict_data[str_x] += 1
            else:
                dict_data[str_x] = 1

        return dict_data

    def _initialize_data_loaders(self, drop_last, num_workers):
        if num_workers is None:
            if isinstance(self.model, nn.DataParallel) and torch.cuda.device_count() > 1:
                num_workers = torch.cuda.device_count()
            else:
                num_workers = 1

        def _init_fn(worker_id):
            np.random.seed(42)
        # num_workers = 1
        # Initializes all data loaders with the loaded datasets
        if hasattr(self.train_dataset, "get_sampler"):
            self.train_data_loader = data.DataLoader(self.train_dataset, batch_sampler=self.train_dataset.get_sampler(self.train_batch_size, drop_last=drop_last), pin_memory=True,
                                                     num_workers=num_workers, worker_init_fn=_init_fn)
            self.val_data_loader = data.DataLoader(self.val_dataset, batch_sampler=self.val_dataset.get_sampler(
                self.train_batch_size, drop_last=False), pin_memory=True, num_workers=1, worker_init_fn=_init_fn)
            self.test_data_loader = data.DataLoader(self.test_dataset, batch_sampler=self.test_dataset.get_sampler(
                self.train_batch_size, drop_last=False), pin_memory=True, num_workers=1, worker_init_fn=_init_fn)
        else:
            self.train_data_loader = data.DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, pin_memory=True, drop_last=drop_last, num_workers=num_workers,
                                                     worker_init_fn=_init_fn)
            self.val_data_loader = data.DataLoader(self.val_dataset, batch_size=self.batch_size,
                                                   shuffle=False, pin_memory=True, drop_last=False, num_workers=1, worker_init_fn=_init_fn)
            self.test_data_loader = data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                                    shuffle=False, pin_memory=True, drop_last=False, num_workers=1, worker_init_fn=_init_fn)

        self.train_data_loader_iter = iter(self.train_data_loader)

    def train_step(self, iteration=0):
        # Check if training data was correctly loaded
        if self.train_data_loader_iter is None:
            print("[!] ERROR: Iterator of the training data loader was None. Additional parameters: " +
                  "train_data_loader was %sloaded, " % ("not " if self.train_data_loader is None else "") +
                  "train_dataset was %sloaded." % ("not " if self.train_dataset is None else ""))

        # Get batch and put it on correct device
        batch = self._get_next_batch()
        batch = TaskTemplate.batch_to_device(batch)

        # Perform task-specific training step
        return self._train_batch(batch, iteration=iteration)

    def sample(self, num_samples, watch_z_t):
        max_size_sample = self.max_samples  # ran out of mem otherwise?
        num_samples_total = 0
        samples_list = []
        for _ in range(math.ceil(num_samples/max_size_sample)):
            if num_samples_total+max_size_sample > num_samples:
                batch_num_samples = num_samples - num_samples_total
            else:

                batch_num_samples = max_size_sample
            num_samples_total += batch_num_samples
            samples = self.model.sample(num_samples=batch_num_samples,
                                        z_in=None,
                                        length=None,
                                        watch_z_t=watch_z_t)
            samples_list.append(samples['x'].cpu().detach().numpy())
        samples = np.array(samples_list)
        samples = samples.reshape(-1, samples.shape[2])

        return samples

    def get_sample_stats(self, samples):

        r20_corr_abs = compute_corr_higher_order(
            samples, self.higher_order_dict_n_abs)
        samples_dict = self.data_to_dict(samples)
        frac_seen_samples = get_frac_overlap(
            self.training_dict, samples_dict)
        return frac_seen_samples, r20_corr_abs

    def evaluate_sample(self, num_samples):

        samples = self.sample(num_samples)

        return self.get_sample_eval(samples)

    def get_sample_eval(self, samples):
        frac_seen_samples, r20_corr_abs = self.get_sample_stats(
            samples)
        overfit_detected = frac_seen_samples > self.frac_val_seen*1.5
        metrics_dict = {'frac_seen_samples': frac_seen_samples,
                        'overfit_detected': overfit_detected}
        for i, r20_abs in enumerate(r20_corr_abs):
            if r20_abs[1] < 0.05:
                metrics_dict['r_20_abs'+str(i+2)] = r20_abs[0]
            else:
                metrics_dict['r_20_abs'+str(i+2)] = -2

        sample_eval = SamplesEvaluation(
            samples, metrics_dict)

        return sample_eval

    def eval(self, data_loader=None, **kwargs):
        # Default: if no dataset is specified, we use validation dataset
        if data_loader is None:
            
            data_loader = self.val_data_loader
        is_test = (data_loader == self.test_data_loader)

        start_time = time.time()
        torch.cuda.empty_cache()
        self.model.eval()

        # Prepare metrics
        nll_counter = 0
        result_batch_dict = {}
        # Evaluation loop
        with torch.no_grad():
            for batch_ind, batch in enumerate(data_loader):

                print("Evaluation process: %4.2f%%" %
                      (100.0 * batch_ind / len(data_loader)), end="\r")
                # Put batch on correct device
                batch = TaskTemplate.batch_to_device(batch)
                # Evaluate single batch
                batch_size = batch[0].size(0) if isinstance(
                    batch, tuple) else batch.size(0)
                batch_dict = self._eval_batch(
                    batch, is_test=is_test)
                for key, batch_val in batch_dict.items():
                    if key in result_batch_dict:
                        result_batch_dict[key] += batch_val.item() * batch_size
                    else:
                        result_batch_dict[key] = batch_val.item() * batch_size

                nll_counter += batch_size

                if self.debug and batch_ind > 10:
                    break
        detailed_metrics = {}
        for key, batch_val in result_batch_dict.items():
            detailed_metrics[key] = batch_val / max(1e-5, nll_counter)

        

        self.model.train()
        eval_time = int(time.time() - start_time)
        print("Finished %s with loss of %4.3f, (%imin %is)" % ("testing" if data_loader ==
                                                              self.test_data_loader else "evaluation", detailed_metrics["loss"], eval_time/60, eval_time % 60))
        torch.cuda.empty_cache()

        
        return detailed_metrics

    def _train_batch(self, batch, iteration):
        x_in, x_length, x_channel_mask = self._preprocess_batch(batch)
        neg_boundnl = -self.model(
            x_in,
            reverse=False,
            get_ldj_per_layer=True,
            beta=self.beta_scheduler.get(iteration),
            length=x_length)

        loss = (neg_boundnl / x_length.float()).mean()
        self.summary_dict["ldj"].append(loss.item())
        self.summary_dict["beta"] = self.beta_scheduler.get(iteration)

        return loss

    def loss_to_bpd(self, loss):
        return (np.log2(np.exp(1)) * loss)

    def test(self, **kwargs):
        return self.eval(data_loader=self.test_data_loader, **kwargs)

    def add_summary(self, writer, iteration, checkpoint_path=None):
        # Adding metrics collected during training to the tensorboard
        # Function can/should be extended if needed
        for key, val in self.summary_dict.items():
            summary_key = "train_%s/%s" % (self.name, key)
            # If it is not a list, it is assumably a single scalar
            if not isinstance(val, list):
                writer.add_scalar(summary_key, val, iteration)
                self.summary_dict[key] = 0.0
            elif len(val) == 0:  # Skip an empty list
                continue
            # For a list of scalars, report the mean
            elif not isinstance(val[0], list):
                writer.add_scalar(summary_key, mean(val), iteration)
                self.summary_dict[key] = list()
            else:  # List of lists indicates a histogram
                val = [v for sublist in val for v in sublist]
                writer.add_histogram(summary_key, np.array(val), iteration)
                self.summary_dict[key] = list()

    def _get_next_batch(self):
        # Try to get next batch. If one epoch is over, the iterator throws an error, and we start a new iterator
        try:
            batch = next(self.train_data_loader_iter)
        except StopIteration:
            self.train_data_loader_iter = iter(self.train_data_loader)
            batch = next(self.train_data_loader_iter)
            self.train_epoch += 1
        return batch

    def _eval_batch(self, batch, is_test=False):
        x_in, x_length, x_channel_mask = self._preprocess_batch(batch)
        ldj = self.model(x_in,
                         reverse=False,
                         get_ldj_per_layer=False,
                         beta=1,
                         length=x_length)

        loss = -(ldj / x_length.float()).mean()
        std_loss = (ldj / x_length.float()).std()

        return {'loss': loss, 'std_loss': std_loss}

    @staticmethod
    def batch_to_device(batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = tuple([b.to(get_device()) for b in batch])
        else:
            batch = batch.to(get_device())
        return batch
