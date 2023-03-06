
import numpy as np
from experiments.synthetic.train import TrainSyntheticModeling
from experiments.synthetic.synthetic_run_config import SyntheticRunConfig
from images.plotting_utils import plot_p2
from images.table_helper import helper_figure, helper_print_table
import math
import random


def print_table_6(result, title, dataset='sort'):
    table_one_metrics = ['nll','p_val','dtv', 'Hel']
    # table_one_metrics = ['Hel', 'dtv', 'dtvpos', 'dtvood', 'p_like',
    #                  'p_rare',  'p_val', 'nll']

    num_metrics = len(table_one_metrics)
    print(
        '\\begin{table*}[h] \\centering \\begin{tabular}{l'+'c'*num_metrics+'} \\toprule')
    # print(
    #     '& \multicolumn{4}{c}{ $\Omega$}  & \\multicolumn{2}{c}{ $\\mathcal{P}$}  &  $\\mathcal{P}^{od}$ \\\\ \\midrule')
    helper_print_table(table_one_metrics, result, title)
    print('\\bottomrule')
    print('\\end{tabular}\\caption{ground truth $K=6$ ' +
          dataset+'}\\end{table*}')
    # table_two_metrics = ['rho2abs', 'rho3abs', 'rho4abs', 'rho5abs']
    # num_metrics = len(table_two_metrics)
    # print()
    # print(
    #     '\\begin{table}[h] \\centering \\begin{tabular}{l'+'c'*num_metrics+'} \\toprule')
    # helper_print_table(table_two_metrics, result, title)
    # print('\\bottomrule')
    # print('\\end{tabular}\\caption{ samples $K=6$ ' + dataset+'}\\end{table}')


def print_other_table(result, title, set_size, dataset):
    table_one_metrics = ['Hel', 'dtv', 'dtvpos', 'dtvood', 'p_like',
                         'p_rare',  'p_val']

    num_metrics = len(table_one_metrics)
    print(
        '\\begin{table*}[h] \\centering \\begin{tabular}{l'+'c'*num_metrics+'} \\toprule')
    print(
        '& \multicolumn{4}{c}{ $\Omega$}  & \\multicolumn{2}{c}{ $\\mathcal{P}$}  &  $\\mathcal{P}^{od}$ \\\\ \\midrule')
    helper_print_table(table_one_metrics, result, title)
    print('\\bottomrule')
    print('\\end{tabular}\\caption{ground truth $K=' +
          str(set_size)+'$ ' +
          dataset+'}\\end{table*}')
    table_two_metrics = ['rho2abs', 'rho3abs', 'rho4abs', 'rho5abs']
    num_metrics = len(table_two_metrics)
    print(
        '\\begin{table}[h] \\centering \\begin{tabular}{l'+'c'*num_metrics+'} \\toprule')
    helper_print_table(table_two_metrics, result, title)
    print('\\bottomrule')
    print('\\end{tabular}\\caption{ samples $K=' +
          str(set_size)+'$ ' + dataset+'}\\end{table}')


def add_dict(smaller_dict, big_dict):
    for key, val in smaller_dict.items():
        if key in big_dict:
            big_dict[key] += val
        else:
            big_dict[key] = val


def compile_result(list_sample_eval, big_sample_eval, detailed_metrics_test, K, S, figure_path=None):
    all_metric_keys = list_sample_eval[0].metrics_dict.keys()
    result_to_print = {}
    for key in all_metric_keys:
        try:
            all_metric = [100*sample_eval.metrics_dict[key]
                          for sample_eval in list_sample_eval if not sample_eval.metrics_dict[key] == -2]
            result_to_print[key] = (np.mean(all_metric), np.std(
                all_metric), 100 * big_sample_eval.metrics_dict[key], all_metric)

        except:
            print('didnt have metric', key)
            print('-------------')
    result_to_print['nll'] = detailed_metrics_test['nll']
    if S == 6 and figure_path is not None:
        check_samples(
            K, S, big_sample_eval.all_dict['histogram_samples_per_p'], figure_path)
    return result_to_print


def check_samples(K, S, histogram, figure_path):
    likely_key = max(histogram.keys())
    list_keys = list(histogram[likely_key].keys())
    list_values = list(histogram[likely_key].values())
    sorted_index = np.argsort(list_values)
    sorted_keys = [list_keys[i] for i in sorted_index]
    for k in sorted_keys[0:10]:
        print(k, histogram[likely_key][k])
    for k in sorted_keys[-10:]:
        print(k, histogram[likely_key][k])
    plot_p2(K, S, histogram, figure_path)


if __name__ == "__main__":
    runconfig = SyntheticRunConfig()
    runconfig.checkpoint_path = 'checkpoints/synthetic_high_K/S_/5/K_/6/CDM/09_06_2022__11_57/cktp/'
    trainModule = TrainSyntheticModeling(runconfig,
                                         batch_size=runconfig.batch_size,
                                         checkpoint_path=runconfig.checkpoint_path,
                                         multi_gpu=runconfig.use_multi_gpu,
                                         silence=False,
                                         path_experiment="")

    eval_aggregate = trainModule.complete_evaluation()
    eval_aggregate.get_average_std()


def run_eval(runconfig):
    trainModule = TrainSyntheticModeling(runconfig,
                                         batch_size=runconfig.batch_size,
                                         checkpoint_path=runconfig.checkpoint_path,
                                         multi_gpu=runconfig.use_multi_gpu,
                                         silence=True,
                                         path_experiment="")

    return trainModule.complete_evaluation(num_trial=10)
