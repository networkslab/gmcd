import sys
import numpy as np
import torch
from multiprocessing import Pool
import os
import time
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import random
import itertools


def to_one_hot(seq, K):
    S = seq.shape[1]
    num_samples = seq.shape[0]
    one_hot = np.zeros((num_samples, S*K))
    for m in range(num_samples):
        for i in range(S):
            one_hot[m, i*K+int(seq[m, i])] = 1
    return one_hot

 # for covariance comp., encode in key pos 1 i pos 2 j and their category


def get_key_from_pos_k1k2(i, j, k_1, k_2):
    pos_key_k1 = get_key_from_pos_k(i, k_1)
    pos_key_k2 = get_key_from_pos_k(j, k_2)
    pos_pair_key = pos_key_k1+':' + pos_key_k2
    return pos_pair_key


def get_key_from_pos_k(i, k):
    return str(int(float(i))) + '_' + str(int(float(k)))


def get_pairwise_freq(seq):  # returns dict i-j_k1-k2
    S = seq.shape[1]
    num_samples = seq.shape[0]
    pairwise_freq_dict = {}
    freq_plus_one = 1/num_samples
    for m in range(num_samples):
        x = seq[m, :]
        for i in range(S):
            for j in range(i+1, S):
                pos_pair_key = get_key_from_pos_k1k2(i, j, x[i], x[j])
                if pos_pair_key in pairwise_freq_dict:
                    pairwise_freq_dict[pos_pair_key]['frequency'] += freq_plus_one
                    pairwise_freq_dict[pos_pair_key]['list_indices'].append(m)
                else:
                    pairwise_freq_dict[pos_pair_key] = {
                        'frequency': freq_plus_one, 'list_indices': [m]}
    return pairwise_freq_dict


def single_pos_freq(seq):
    S = seq.shape[1]
    num_samples = seq.shape[0]
    freq_plus_one = 1/num_samples
    freq_single_pos = {}
    for m in range(num_samples):
        for i in range(S):
            x = int(seq[m, i])
            pos_key = get_key_from_pos_k(i, x)
            if pos_key in freq_single_pos:
                freq_single_pos[pos_key] += freq_plus_one
            else:
                freq_single_pos[pos_key] = freq_plus_one

    return freq_single_pos


def get_list_covariance(seq, K):
    S = seq.shape[1]
    freq_single_pos = single_pos_freq(seq)
    pairwise_freq_dict = get_pairwise_freq(seq)
    dict_cov = {}
    for i in range(S):
        for j in range(i+1, S):
            for k_1 in range(K):
                # here we can check if we have i with k_1, if not it all be zeros
                pos_key_k1 = get_key_from_pos_k(i, k_1)
                if pos_key_k1 in freq_single_pos:
                    f_k1_i = freq_single_pos[pos_key_k1]

                    for k_2 in range(K):
                        pos_key_k2 = get_key_from_pos_k(j, k_2)
                        if pos_key_k2 in freq_single_pos:
                            f_k2_j = freq_single_pos[pos_key_k2]
                            pairwise_freq = 0
                            indices = []
                            pos_pair_key = get_key_from_pos_k1k2(
                                i, j, k_1, k_2)
                            if pos_pair_key in pairwise_freq_dict:
                                pairwise_freq = pairwise_freq_dict[pos_pair_key]['frequency']
                                indices = pairwise_freq_dict[pos_pair_key]['list_indices']
                            cov = pairwise_freq - f_k1_i * f_k2_j
                            dict_cov[pos_key_k1+':' +
                                     pos_key_k2] = {'cov': cov, 'list_indices': indices}
                        else:
                            dict_cov[pos_key_k1+':' +
                                     pos_key_k2] = {'cov': 0, 'list_indices': []}
                else:
                    for k_2 in range(K):
                        pos_key_k2 = get_key_from_pos_k(j, k_2)
                        dict_cov[pos_key_k1+':' +
                                 pos_key_k2] = {'cov': 0, 'list_indices': []}

    # dict[s_k1:t_k2] = 0 or (pairwise_freq - f_k1_i * f_k2_j, [index])
    return dict_cov


def plot_side(sorted_list_gt, sorted_diff, filepath, xlabel):
    pos_diff = []
    neg_diff = []
    for i, d in enumerate(sorted_diff):
        if d < 0:
            neg_diff.append(i)
        else:
            pos_diff.append(i)

    pos_error_total = np.sum(sorted_diff[pos_diff])
    neg_error_total = np.sum(sorted_diff[neg_diff])
    plt.plot(sorted_list_gt, 'k', alpha=0.5)
    plt.vlines(neg_diff, ymin=0, ymax=sorted_diff[neg_diff], color='r',
               label='underestimated cov. {:.1f}'.format(neg_error_total))
    plt.vlines(pos_diff, ymin=0, ymax=sorted_diff[pos_diff], color='g',
               label='overestimated cov. {:.1f}'.format(pos_error_total))
    plt.xlabel(xlabel)
    plt.ylabel('Ground truth covariance with sample error')
    plt.legend()

    plt.savefig(filepath)
    plt.close()


def plot_corr(ground_truth, diff_dict, figure_path):

    list_gt = np.array(list(ground_truth.values()))
    list_diff_dict = np.array(list(diff_dict.values()))
    index = np.argsort(list_gt)
    return
    sorted_list_gt = list_gt[index]
    sorted_diff = list_diff_dict[index]
    print(sorted_list_gt)
    pos_cov_index = [i for i, v in enumerate(sorted_list_gt) if v > 0][0]
    os.path.join(figure_path, 'cov_neg.pdf')
    plot_side(sorted_list_gt[:pos_cov_index], sorted_diff[:pos_cov_index], os.path.join(
        figure_path, 'cov_neg.pdf'), 'Every possible pair in the sequence with negative cov.')
    plot_side(sorted_list_gt[pos_cov_index:], sorted_diff[pos_cov_index:], os.path.join(
        figure_path, 'cov_pos.pdf'), 'Every possible pair in the sequence with positive cov.')


if __name__ == '__main__':
    # file_sample = 'sample_cov.pk'
    # file_gt = 'ground_truth_cov.pk'
    # with open(file_gt, 'rb') as f:
    #     dict_cov_val = pk.load(f)
    # with open(file_sample, 'rb') as f:
    #     dict_cov = pk.load(f)

    # X = []
    # Y = []
    # diff_dict = {}
    # ground_truth = {}
    # for key in dict_cov_val.keys():  # make sure that the two lists are aligned
    #     X.append(dict_cov[key]['cov'])
    #     Y.append(dict_cov_val[key]['cov'])
    #     ground_truth[key] = dict_cov_val[key]['cov']
    #     diff_dict[key] = dict_cov_val[key]['cov'] - dict_cov[key]['cov']
    plot_corr()


def compute_pairwise_corr(dict_cov, dict_cov_val, figure_path):

    X = []
    Y = []
    diff_dict = {}
    ground_truth = {}
    for key in dict_cov_val.keys():  # make sure that the two lists are aligned
        X.append(dict_cov[key]['cov'])
        Y.append(dict_cov_val[key]['cov'])
        ground_truth[key] = dict_cov_val[key]['cov']
        diff_dict[key] = dict_cov_val[key]['cov'] - dict_cov[key]['cov']
    plot_corr(ground_truth, diff_dict, figure_path)
    return pearsonr(X, Y)


"""
Higher order covariance
"""


def sample_pos_of_pattern_n(p, S, num_patterns):
    if S < 20:
        all_pos_of_pattern = list(
            set(itertools.combinations([i for i in range(S)], p)))
        num_pos_of_pattern = len(all_pos_of_pattern)
        pos_of_pattern_indices = list(range(num_pos_of_pattern))

        def index_to_pos_of_pattern(index):  # maps ind -> [0,2,3,4]
            return all_pos_of_pattern[index]

        if num_pos_of_pattern < num_patterns:  # just return all of them
            samples_pos = [sorted(index_to_pos_of_pattern(
                s_index)) for s_index in pos_of_pattern_indices]
            return samples_pos

        samples_index = random.sample(pos_of_pattern_indices, num_patterns)
        samples_pos = [sorted(index_to_pos_of_pattern(
            s_index)) for s_index in samples_index]

        return samples_pos
    else:

        positions = list(range(S))
        dict_pos = {}
        retries = 0

        def list_to_key(list1):
            list1.sort()
            return '-'.join([str(i) for i in list1])
        for _ in range(num_patterns):

            d = list_to_key(random.sample(positions, p))
            while d in dict_pos:
                retries += 1
                d = list_to_key(random.sample(positions, p))
            dict_pos[d] = 1
        samples_pos = [[int(i)for i in pos.split('-')]
                       for pos in list(dict_pos.keys())]
        return samples_pos


def cat_to_key(x):
    key = str(x[0])
    for s in x[1:]:
        key = key+'-'+str(s)
    return key


def pattern_to_freq_dict(pattern_n_seq, freq_plus_one=None):
    num_samples = pattern_n_seq.shape[0]
    if freq_plus_one is None:
        freq_plus_one = 1/num_samples
    dict_word = {}
    for m in range(num_samples):
        x = pattern_n_seq[m, :]
        key = cat_to_key(x)
        if key in dict_word:
            dict_word[key] += freq_plus_one
        else:
            dict_word[key] = freq_plus_one
    return dict_word


def find_top_20(pos, seq, absolute_version):  # find top 20 at those positions
    pattern_n_seq = seq[:, pos]
    dict_word = pattern_to_freq_dict(pattern_n_seq)
    dict_top_20 = {}
    min = (None, 2)
    for key, val in dict_word.items():
        abs_val = val
        if absolute_version:
            abs_val = np.abs(val)
        if len(dict_top_20.values()) < 20:
            dict_top_20[key] = val
            if abs_val < min[1]:
                min = (key, abs_val)
        else:
            if abs_val < min[1]:
                del dict_top_20[min[0]]
                min = (key, abs_val)
                dict_top_20[key] = val

    return dict_top_20


def mp_func(pos, seq, freq_single_pos):
    top_20_freq_for_pos = find_top_20(
        pos, seq, absolute_version=True)
    higher_order_cov = []
    for word, freq in top_20_freq_for_pos.items():
        C_word = freq
        f_i_k = 1
        for i, k in enumerate(word.split('-')):
            pos_of_i = pos[i]
            pos_key = get_key_from_pos_k(pos_of_i, k)
            f_i_k *= freq_single_pos[pos_key]
        cov = C_word - f_i_k
        higher_order_cov.append(cov)
    time.sleep(1)
    return higher_order_cov, top_20_freq_for_pos


def compute_higher_order_stats(seq, num_patterns, n, absolute_version=False):

    S = seq.shape[1]
    freq_single_pos = single_pos_freq(seq)  # f_i_k

    all_pos = sample_pos_of_pattern_n(
        n, S, num_patterns)  # sample random pattern

    with Pool(8) as p:
        mp_result = p.starmap(
            mp_func, [(pos, seq, freq_single_pos) for pos in all_pos])

    higher_order_cov_list_of_list = [result[0] for result in mp_result]
    higher_order_cov = [
        item for sublist in higher_order_cov_list_of_list for item in sublist]
    top_20_freq = [result[1] for result in mp_result]

    higher_order_dict = {
        'all_pos': all_pos, 'top_20_freq': top_20_freq, 'higher_order_cov': higher_order_cov}
    return higher_order_dict


def get_freq_dict_of_pos(pos, samples, words):
    pattern_n_seq = samples[:, pos]
    dict_word = pattern_to_freq_dict(pattern_n_seq)
    top_20_freq_for_pos = {}
    for word in words:
        if word in dict_word:
            top_20_freq_for_pos[word] = dict_word[word]
        else:
            top_20_freq_for_pos[word] = 0
    return top_20_freq_for_pos


# with Pool(8) as p:
#         mp_result = p.starmap(
#             mp_func, [(pos, seq, freq_single_pos) for pos in all_pos])

def mp_cov_func(i, pos, samples, higher_order_dict, freq_single_pos):
    higher_order_cov_ind = []
    top_20_freq_for_pos_baseline = higher_order_dict['top_20_freq'][i]
    top_20_freq_for_pos = get_freq_dict_of_pos(
        pos, samples, top_20_freq_for_pos_baseline.keys())
    for word, freq in top_20_freq_for_pos.items():
        C_word = freq
        f_i_k = 1
        for i, k in enumerate(word.split('-')):
            pos_of_i = pos[i]
            pos_key = get_key_from_pos_k(pos_of_i, k)
            if pos_key not in freq_single_pos:
                f_i_k = 0
                break
            else:
                f_i_k *= freq_single_pos[pos_key]

        cov = C_word - f_i_k
        higher_order_cov_ind.append(cov)
    time.sleep(1)
    return higher_order_cov_ind


def compute_corr_higher_order(samples, higher_order_dict_n):
    r20_n = []
    freq_single_pos = single_pos_freq(samples)  # f_i_k
    for _, higher_order_dict in higher_order_dict_n.items():
        all_pos = higher_order_dict['all_pos']

        with Pool(8) as p:
            # schedule one map/worker for each row in the original data
            higher_order_cov_mp_list_of_list = p.starmap(
                mp_cov_func, [(i, pos, samples, higher_order_dict, freq_single_pos) for i, pos in enumerate(all_pos)])
        higher_order_cov_mp = [
            item for sublist in higher_order_cov_mp_list_of_list for item in sublist]

        higher_order_cov_baseline = higher_order_dict['higher_order_cov']
        mp_pearsonr = pearsonr(higher_order_cov_mp, higher_order_cov_baseline)

        r20_n.append(mp_pearsonr)

    return r20_n


"""
Hamming helper
"""


def get_hamming_dist_distribution(seq):
    num_samples = seq.shape[0]
    num_samples = min(num_samples, 1000)
    freq_plus_one = 2/(num_samples * (num_samples-1))
    hamming_dist_distribution = {}
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            d = get_hamming_distance(seq[i], seq[j])
            if d in hamming_dist_distribution:
                hamming_dist_distribution[d] += freq_plus_one
            else:
                hamming_dist_distribution[d] = freq_plus_one
    return hamming_dist_distribution


def get_hamming_distance(seq_1, seq_2):
    S = seq_1.shape[0]
    d = 0
    for s in range(S):
        if seq_1[s] != seq_2[s]:
            d += 1
    return d


def get_total_var_dist(dict_support_1, dict_support_2):
    tv = 0
    remaining_support_dict_2 = list(dict_support_2.keys())
    for key, val in dict_support_1.items():
        if key in dict_support_2:
            diff = np.abs(val - dict_support_2[key])
            remaining_support_dict_2.remove(key)
        else:
            diff = val  # support of dict 2 at that pos is 0

        tv += diff

    for key in remaining_support_dict_2:  # support of dict 1 at that pos is 0
        diff = dict_support_2[key]
        tv += diff
    return tv/2


def one_hot(x, num_classes, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        x_onehot = np.zeros(x.shape + (num_classes,), dtype=np.float32)
        x_onehot[np.arange(x.shape[0]), x] = 1.0
    elif isinstance(x, torch.Tensor):
        assert torch.max(
            x) < num_classes, "[!] ERROR: One-hot input has larger entries (%s) than classes (%i)" % (str(torch.max(x)), num_classes)
        x_onehot = x.new_zeros(x.shape + (num_classes,), dtype=dtype)
        x_onehot.scatter_(-1, x.unsqueeze(dim=-1), 1)
    else:
        print("[!] ERROR: Unknown object given for one-hot conversion:", x)
        sys.exit(1)
    return x_onehot


def get_frac_overlap(training_dict, samples_dict):
    frac_overlap = 0
    frac_non_overlap = 0
    freq = 1/np.sum(list(samples_dict.values()))
    for key, val in samples_dict.items():
        if key in training_dict:
            frac_overlap += val * freq
        else:
            frac_non_overlap += val*freq
    return frac_overlap


def get_exact_metrics(exact_prob, K, S, M):
    size_pos_support = np.prod([K-i for i in range(S)])
    C = 1 / size_pos_support
    p_likely = 3*C/2
    p_rare = C/2

    d_tv = hellinger = tv_ood = exact_kl = 0

    def get_p_key(seq):
        dict_seq = {}
        for x in seq:
            if x in dict_seq:
                return 0
            else:
                dict_seq[x] = 1
        if seq[0] < seq[-1]:
            return p_likely
        else:
            return p_rare

    for key, p_emp in exact_prob.items():
        p = get_p_key(key)
        if p == 0:
            d_tv += p_emp
            tv_ood += p_emp
            hellinger += p_emp
        else:
            exact_kl += p*(np.log(p) - np.log(p_emp))
            d_tv += np.abs(p - p_emp)
            hellinger += (np.sqrt(p) - np.sqrt(p_emp))**2
    hellinger = np.sqrt(0.5)*np.sqrt(hellinger)
    d_tv = 0.5*d_tv
    tv_ood = 0.5*tv_ood
    return d_tv, hellinger, tv_ood, exact_kl


def get_diff_metric(histogram_samples_per_p, size_support_dict, M):
    d_tv = hellinger = tv_ood = 0

    for p, dict_p in histogram_samples_per_p.items():
        size_support_p = size_support_dict[p]
        list_val = dict_p.values()
        i = 0
        for i, count in enumerate(list_val):
            empirical_p_val = count / M
            d_tv += np.abs(p - empirical_p_val)
            if p == 0:
                tv_ood += np.abs(p - empirical_p_val)
            hellinger += (np.sqrt(p) - np.sqrt(empirical_p_val))**2
        d_tv += (size_support_p-i)*p
        hellinger += (size_support_p-i)*p

    hellinger = np.sqrt(0.5)*np.sqrt(hellinger)
    d_tv = 0.5*d_tv
    tv_ood = 0.5*tv_ood
    return d_tv, hellinger, tv_ood


def KL(based_dist, to_dist):  # support of based must cover to_dist
    KL = 0
    floor_p = 1/len(based_dist.keys())
    for key, val in based_dist.items():
        if key not in to_dist:
            q = floor_p  # basically - inf
        else:
            q = to_dist[key]
        p = val+floor_p
        KL += p * (np.log(p) - np.log(q))
    return KL


def zero_support(dict_invalid, num_samples):
    zero_entropy = 0
    for _, val in dict_invalid.items():
        p = val/num_samples
        zero_entropy += p * np.log(p)
    return -zero_entropy


def generalization_metric(training_set_permu, histogram_samples_per_p):
    count_val = 0
    count_total = 0
    freq_train = 0
    num_training_examples = sum(training_set_permu.values())
    for key, val_dicts in histogram_samples_per_p.items():
        if key > 0:
            for key, val in val_dicts.items():
                count_total += val
                if key not in training_set_permu:
                    count_val += val
                else:
                    freq_train += val * \
                        training_set_permu[key]/num_training_examples

    if count_total == 0:
        return 0, 0
    return count_val/count_total, freq_train


def one_error(key):
    tokens = key.split('-')
    s = len(tokens)
    dict_in = {}
    for i in tokens:
        if i in dict_in:
            if dict_in[i] == 1:
                dict_in[i] += 1
            else:
                return False
        else:
            dict_in[i] = 1
    if len(dict_in.keys()) == s:
        return False
    return True
