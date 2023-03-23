from src.optimizer.Scheduler import ExponentialScheduler
from src.datasets.synthetic import SyntheticDataset
from src.metrics import generalization_metric, get_diff_metric, get_exact_metrics
from src.sample_output import SamplesEvaluation
from src.task_template import TaskTemplate
import torch


class TaskSyntheticModeling(TaskTemplate):
    def __init__(self,
                 model,
                 run_config,
                 load_data=True,
                 debug=False,
                 batch_size=64):
        super().__init__(model,
                         run_config,
                         load_data=load_data,
                         debug=debug,
                         batch_size=batch_size,
                         name="TaskSyntheticModeling")
        self.beta_scheduler = self.create_scheduler(self.run_config)

        self.summary_dict = {
            "log_prob": list(),
            "ldj": list(),
            "z": list(),
            "beta": 0
        }

    def create_scheduler(self, scheduler_params, param_name=None):
        end_val = scheduler_params.beta_scheduler_end_val
        start_val = scheduler_params.beta_scheduler_start_val
        stepsize = scheduler_params.beta_scheduler_step_size
        logit = scheduler_params.beta_scheduler_logit
        delay = scheduler_params.beta_scheduler_delay

        return ExponentialScheduler(start_val=start_val, end_val=end_val, logit_factor=logit, stepsize=stepsize, delay=delay, param_name=param_name)

    def _load_datasets(self):
        self.S = self.run_config.S
        self.K = self.run_config.K
        dataset_name = self.run_config.dataset
        dataset_class, dataset_kwargs = TaskSyntheticModeling.get_dataset_class(
            dataset_name, return_kwargs=True)
        print("Loading dataset %s..." % dataset_name)

        self.train_dataset = dataset_class(S=self.S, K=self.K,
                                           train=True,
                                           **dataset_kwargs)
        self.val_dataset = dataset_class(S=self.S, K=self.K,
                                         val=True,
                                         **dataset_kwargs)
        self.test_dataset = dataset_class(S=self.S, K=self.K,
                                          test=True,
                                          **dataset_kwargs)

    @staticmethod
    def get_dataset_class(dataset_name, return_kwargs=False):
        dataset_kwargs = {'dataset_name': dataset_name}
        dataset_class = SyntheticDataset

        if return_kwargs:
            return dataset_class, dataset_kwargs
        else:
            return dataset_class

    def evaluate_sample(self,num_samples):  # to do batch samples
        self.base_sample_eval = super().evaluate_sample(num_samples)
        return self.get_sample_eval(self.base_sample_eval)

    def get_sample_eval(self, samples):
        m = samples.shape[0] # num samples
        histogram_samples_per_p = self.train_dataset.samples_to_dict(samples)
        size_support_dict = self.train_dataset.get_size_support_dict()
        empirical_prob = {}
        for _, dict_p in histogram_samples_per_p.items():
            for key, val in dict_p.items():
                empirical_prob[key] = val / m
        # unseen_support, train_slice = generalization_metric(
        #     self.train_dataset.dict_permu, histogram_samples_per_p)

        all_ps = list(histogram_samples_per_p.keys())
        if 0 in all_ps:
            all_ps.remove(0)
        if len(all_ps) == 0:
            p_likely = p_rare = 0
        else:
            p_likely = sum(histogram_samples_per_p[max(
                all_ps)].values()) / m
            p_rare = sum(histogram_samples_per_p[min(
                all_ps)].values()) / m
        p_total = p_likely+p_rare
        metrics_dict = {'p_rare': p_rare, 'p_likely': p_likely, 'p_total': p_total}
        
        metrics_dict['d_tv'], metrics_dict['hellinger'], metrics_dict['tv_ood'] = get_diff_metric(
            histogram_samples_per_p, size_support_dict, M=m)

        metrics_dict.update(samples.metrics_dict)
        sample_eval = SamplesEvaluation(
            samples, metrics_dict, {'histogram_samples_per_p': histogram_samples_per_p})
        return sample_eval

    def _train_batch_discrete(self, x_in, x_length):
        _, ldj = self.model(x_in, reverse=False, length=x_length)
        loss = (-ldj / x_length.float()).mean()
        return loss

    def _calc_loss(self, neg_ldj, neglog_prob, x_length, take_mean=True):
        neg_ldj = (neg_ldj / x_length.float())
        neglog_prob = (neglog_prob / x_length.float())
        loss = neg_ldj + neglog_prob
        if take_mean:
            loss = loss.mean()
            neg_ldj = neg_ldj.mean()
            neglog_prob = neglog_prob.mean()
        return loss, neg_ldj, neglog_prob

    def _preprocess_batch(self, batch):
        x_in = batch
        x_length = x_in.new_zeros(x_in.size(0),
                                  dtype=torch.long) + x_in.size(1)
        x_channel_mask = x_in.new_ones(x_in.size(0),
                                       x_in.size(1),
                                       1,
                                       dtype=torch.float32)
        return x_in, x_length, x_channel_mask

    def initialize(self, num_batches=16):

        if self.model.need_data_init():
            # print("Preparing data dependent initialization...")
            batch_list = []
            for _ in range(num_batches):
                batch = self._get_next_batch()
                batch = TaskTemplate.batch_to_device(batch)
                x_in, x_length, _ = self._preprocess_batch(batch)
                batch_tuple = (x_in, {"length": x_length})
                batch_list.append(batch_tuple)
            self.model.initialize_data_dependent(batch_list)
