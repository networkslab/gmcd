import numpy as np
import matplotlib.pyplot as plt

MAX_E = 100000


def distance_l2(x, y):
    return np.linalg.norm(x - y)


def distance_cos(x, y):
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    return np.arccos(np.clip(np.dot(x, y), -1.0, 1.0))


def normalize_means(means):
    norm = np.linalg.norm(means, axis=1)
    norm = np.power(norm, -1).reshape(-1, 1)
    us = means * norm
    return us


def compute_min_dist(means, dis):
    all_points = means.shape[0]
    dist = []
    for i in range(all_points - 1):
        for j in range(i + 1, all_points):
            d = dis(means[i, :], means[j, :])
            dist.append(d)
    min_dist = np.min(dist)
    return min_dist, dist


def perturbation(us, on_circle):

    K = us.shape[0]
    d = us.shape[1]
    i = np.random.choice(K)
    if on_circle:
        us_prime = np.array(us, copy=True)
        norm = np.linalg.norm(us_prime, axis=1)
        us_prime[i, :] = us_prime[i, :] + 0.1 * (np.random.rand(1, d) - 0.5)
        us_prime = normalize_means(us_prime)
    else:
        us_prime = np.array(us, copy=True)
        us_prime[i, :] = us_prime[i, :] + 0.1 * (np.random.rand(1, d) - 0.5)
        us_prime = us_prime.clip(-1, 1)
    return us_prime


def energy(us, dist):
    _, dist = compute_min_dist(us, dist)
    dist = np.array(dist)
    if not np.all(dist):
        return None
    E = np.sum(np.power(dist, -2))
    return E


def get_mean_wrapper(K, d, not_doing):
    np.random.seed(2)
    means = np.random.rand(K, d) * 2 - 1
    on_circle = True
    if on_circle:
        means = normalize_means(means)

    if not_doing:
        print('NOT COMPUTING THE MEANS')
        us_final = means
    else:

        def perturbation_l2(x): return perturbation(x, on_circle=on_circle)
        us_final, _ = annealed_temperature(means, distance_l2,
                                           perturbation_l2,
                                           view=False)

    us_final = us_final*2
    min_dist, _ = compute_min_dist(us_final, distance_l2)

    return us_final, min_dist


def view_means(init_means, means):
    var_value = 0.1
    circles = []
    for i in range(means.shape[0]):

        circles.append(
            plt.Circle((means[i, 0], means[i, 1]), var_value, color='b'))
    for i in range(init_means.shape[0]):
        circles.append(
            plt.Circle((init_means[i, 0], init_means[i, 1]),
                       var_value,
                       color='r'))
    _, ax = plt.subplots()
    for c in circles:
        ax.add_patch(c)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()
    plt.close()


def annealed_temperature(us_init, dist, perturbation, view):
    T = 10
    alpha = 0.9
    COUNT_MAX = 100
    MIN_DELTA_FACTOR = 0.001  # for other
    # MIN_DELTA_FACTOR = 0.01 # for text
    notdone = True
    count = 0
    store_delta = []
    store_p = []
    track_jump = []
    Ts = []
    us = np.array(us_init, copy=True)
    while notdone:
        us_prime = perturbation(us)
        E_prime = energy(us_prime, dist)

        while E_prime is None:  # if step made it too close to another point
            us_prime = perturbation(us_prime)
            E_prime = energy(us_prime, dist)

        delta_E = E_prime - energy(us, dist)  # if us > us_prime

        store_delta.append(np.abs(delta_E))
        if delta_E < 0:
            us = us_prime
            track_jump.append(1)

        else:
            track_jump.append(0)
            p = np.exp(-delta_E / T)
            store_p.append(p)
            coin_flips = np.random.binomial(1, p)
            if coin_flips == 1:
                us = us_prime
        count += 1
        Ts.append(T)
        if count > COUNT_MAX:

            T = T * alpha
            count = 0

        if len(store_delta) > 500 and (
                np.mean(store_delta[-5:-1]) <
                MIN_DELTA_FACTOR * np.mean(store_delta[0:500])
                or np.std(store_delta[-5:-1]) < 0.0001):
            if view:
                plt.plot(track_jump, 'o', label='jump')
                plt.plot(store_delta, label='delta')
                plt.plot(Ts, label='T')

                plt.legend()
                plt.show()
                plt.close()

            notdone = False
    return us, us_init


def cos_annealed_temperature(K, d, view=False):

    means = np.random.rand(K, d) * 2 - 1
    us_init = normalize_means(means)
    def perturbation_cos(x): return perturbation(x, on_circle=True)
    return annealed_temperature(us_init, distance_cos, perturbation_cos, view)


def random(K, d):
    means = np.random.rand(K, d) * 2 - 1
    return means


if __name__ == '__main__':
    # K = 27
    # d = 3
    # on_circle=True

    K = 27
    d = 2
    on_circle = False

    means = np.random.rand(K, d) * 2 - 1
    if on_circle:
        means = normalize_means(means)

    def perturbation_l2(x): return perturbation(x, on_circle=on_circle)
    us_final, us_init = annealed_temperature(means,
                                             distance_l2,
                                             perturbation_l2,
                                             view=True)
    view_means(means, us_final)
    print('started at')
    min_dist, dist = compute_min_dist(means, distance_l2)
    print('min distance between circles', min_dist)
    print('mean distance between circles', np.mean(dist))
    print('ended at')
    min_dist, dist = compute_min_dist(us_final, distance_l2)
    print('min distance between circles', min_dist)
    print('mean distance between circles', np.mean(dist))
