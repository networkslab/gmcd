import numpy as np
import math
import random


def transform_to_mgrain(aggregated_dict_valid_rare, aggregated_dict_valid_likely, m,  U):
    list_p_emp = []
    for key, val in aggregated_dict_valid_likely.items():
        random_list = [random.randint(0, 2) for _ in range(val)]
        list_p_emp.append(random_list.count(0)/m)
        list_p_emp.append(random_list.count(1)/m)
        list_p_emp.append(random_list.count(2)/m)
    for key, val in aggregated_dict_valid_rare.items():
        list_p_emp.append(val/m)
    for _ in range(U-len(list_p_emp)):
        list_p_emp.append(0)
    return list_p_emp


def get_expected_S(m, U):
    E = 0
    start_k = math.ceil(m/U)
    for k in range(start_k, m):
        add = (k-m/U) * (1-1/U)**(m-k)/U**k
        E += add
        if add < 1e-10:
            break
    E = U*E
    return E


def get_S_stat(aggregated_dict_valid_likely, aggregated_dict_valid_rare, m, K, S):

    delta = 0.05
    U = 2*math.factorial(K)
    p_uni = 1/U
    smallest_epsilon = np.sqrt(
        (np.sqrt(np.log(1/delta)*U) + np.log(1/delta))/m)
    epsilon = smallest_epsilon
    list_p_emp = transform_to_mgrain(
        aggregated_dict_valid_rare, aggregated_dict_valid_likely, m,  U)
    S = 0
    for p_emp in list_p_emp:
        S += np.abs(p_uni-p_emp)
    S = 0.5*S

    Exp_S = get_expected_S(m, U)
    done = False
    while not done:
        if m < U:
            t = Exp_S + (epsilon*m/U)**2
        elif m < U/epsilon**2:
            t = Exp_S + epsilon**2*np.sqrt(m/U)
        else:
            t = t = Exp_S + epsilon

        if t < S:
            epsilon += (1-smallest_epsilon)/epsilon
        else:
            done = True

    return smallest_epsilon, epsilon, S


if __name__ == '__main__':
    K = 3  # number of categories
    S = 3  # lenght of sequence

    aggregated_dict_valid_likely = {
        'likely_sequence_1': 2, 'likely_sequence_2': 3, 'likely_sequence_3': 2}
    aggregated_dict_valid_rare = {
        'rare_sequence_1': 1, 'rare_sequence_2': 1, 'rare_sequence_3': 2}
    m = np.sum(list(aggregated_dict_valid_likely.values())) + \
        np.sum(list(aggregated_dict_valid_rare.values()))

    print('Number of samples', m)
    smallest_epsilon, epsilon, S = get_S_stat(aggregated_dict_valid_likely,
                                              aggregated_dict_valid_rare, m, K, S)
    print(smallest_epsilon)
    print(epsilon)
    print(S)
