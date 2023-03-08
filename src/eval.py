
import numpy as np


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

    return result_to_print
