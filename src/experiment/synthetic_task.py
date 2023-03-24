from src.sample_metrics import SamplesMetrics
from src.optimizer.scheduler import ExponentialScheduler
from src.datasets.synthetic import SyntheticDataset
from src.metrics import get_diff_metric
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
        print("Loading synthetic dataset K=%s..." % self.S)

        self.train_dataset = SyntheticDataset(S=self.S, K=self.K,
                                              train=True)
        self.val_dataset = SyntheticDataset(S=self.S, K=self.K,
                                            val=True)
        self.test_dataset = SyntheticDataset(S=self.S, K=self.K,
                                             test=True)

    def evaluate_sample(self, num_samples):
        samples_np = self.sample(num_samples)  # obtain the samples
        # compute the metrics on the samples
        return self.get_sample_metrics(samples_np)

    def get_sample_metrics(self, samples_np):
        m = samples_np.shape[0]  # num samples
        base_samples_results = super().get_sample_metrics(samples_np)
        histogram_samples_per_p = self.train_dataset.samples_to_dict(
            samples_np)
        size_support_dict = self.train_dataset.get_size_support_dict()
        empirical_prob = {}
        for _, dict_p in histogram_samples_per_p.items():
            for key, val in dict_p.items():
                empirical_prob[key] = val / m

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

        d_tv, H, tv_ood = get_diff_metric(
            histogram_samples_per_p, size_support_dict, M=m)

        metrics = {'p_rare': p_rare,
                   'p_likely': p_likely, 'p_total': p_total, 'd_tv': d_tv, 'H': H, 'tv_ood': tv_ood, 'histogram_samples_per_p': histogram_samples_per_p}

        sample_results = SamplesMetrics(
            samples_np, metrics, histogram_samples_per_p)
        sample_results.add_new_metrics(base_samples_results.metrics)
        return sample_results

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
