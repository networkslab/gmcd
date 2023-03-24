import sys
import numpy as np
import torch
from multiprocessing import Pool
import time
from scipy.stats.stats import pearsonr
import random
import itertools



def get_key_from_pos_k1k2(i, j, k_1, k_2):
    pos_key_k1 = get_key_from_pos_k(i, k_1)
    pos_key_k2 = get_key_from_pos_k(j, k_2)
    pos_pair_key = pos_key_k1+':' + pos_key_k2
    return pos_pair_key


def get_key_from_pos_k(i, k):
    return str(int(float(i))) + '_' + str(int(float(k)))



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



