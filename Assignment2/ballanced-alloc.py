import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

def make_checkpoints(m: int, num_points: int = 200): # checkpoints for faster computations
    n_max = m * m

    multiples = np.arange(m, n_max + 1, m)
    pts = np.unique(np.geomspace(1, n_max, num=num_points, dtype=int))
    pts = np.concatenate(([1, n_max], pts, multiples))
    pts = np.unique(pts)
    pts.sort()
    return pts


def d_choice(bins, rng, d):
    m = bins.size
    idx = rng.integers(0, m, size=d)
    loads = bins[idx]
    best = loads.min()
    options = np.where(loads == best)[0]
    return int(idx[rng.choice(options)])

def one_choice(bins, rng):
    return d_choice(bins, rng, d=1)

def two_choice(bins, rng):
    return d_choice(bins, rng, d=2)

def one_plus_beta_choice(bins, rng, beta=0.5):
    if rng.random() < beta:
        return two_choice(bins, rng)
    else:
        return one_choice(bins, rng)

def partial_k1(bins, rng):
    m = bins.size
    c1, c2 = rng.integers(0, m, size=2)
    med = np.median(bins)
    a1 = bins[c1] >= med
    a2 = bins[c2] >= med
    if a1 and not a2:  # Q1: > than median?
        return c2
    if a2 and not a1:
        return c1
    return rng.choice([c1, c2])

def partial_k2(bins, rng):
    m = bins.size
    c1, c2 = rng.integers(0, m, size=2)
    med = np.median(bins)
    a1 = bins[c1] >= med
    a2 = bins[c2] >= med

    if a1 and not a2:  # Q1: > than median?
        return c2
    if a2 and not a1:
        return c1

    if not a1 and not a2:
        q25 = np.percentile(bins, 25)
        b1 = bins[c1] >= q25
        b2 = bins[c2] >= q25
        if b1 and not b2:  # Q2: > than 25th centile?
            return c2
        if b2 and not b1:
            return c1
        return rng.choice([c1, c2])
    else:
        q75 = np.percentile(bins, 75)
        b1 = bins[c1] >= q75
        b2 = bins[c2] >= q75
        if b1 and not b2:  # Q2: > than 75th centile?
            return c2
        if b2 and not b1:
            return c1
        return rng.choice([c1, c2])


def run_sequential(m, n_max, rng, checkpoints, alloc_func):
    bins = np.zeros(m, dtype=int)
    res = np.empty(len(checkpoints))
    cp_i = 0
    next_cp = checkpoints[cp_i]

    for n in range(1, n_max + 1):
        chosen = alloc_func(bins, rng)
        bins[chosen] += 1

        if n == next_cp:
            res[cp_i] = bins.max() - n / m
            cp_i += 1
            if cp_i >= len(checkpoints):
                break
            next_cp = checkpoints[cp_i]
    return res

def run_batched(m, n_max, b, rng, checkpoints, alloc_func):
    bins = np.zeros(m, dtype=int)
    res = np.empty(len(checkpoints))
    cp_i = 0
    next_cp = checkpoints[cp_i]
    total = 0

    while total < n_max:
        snapshot = bins.copy()
        batch = min(b, n_max - total)
        choices = [alloc_func(snapshot, rng) for _ in range(batch)]

        for ch in choices:
            bins[ch] += 1
            total += 1

            if total == next_cp:
                res[cp_i] = bins.max() - total / m
                cp_i += 1
                if cp_i >= len(checkpoints):
                    break
                next_cp = checkpoints[cp_i]
        if cp_i >= len(checkpoints):
            break
    return res


# Experiments
def experiment_sequential(T, m, checkpoints, seed, alloc_func):
    n_max = m * m
    rng_master = np.random.default_rng(seed)
    all_res = np.empty((T, len(checkpoints)))

    for t in trange(T, desc="Sequential"):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32)))
        all_res[t] = run_sequential(m, n_max, rng, checkpoints, alloc_func)

    return all_res.mean(axis=0), all_res.std(axis=0)

def experiment_batched(T, m, checkpoints, seed, alloc_func, b):
    n_max = m * m
    rng_master = np.random.default_rng(seed)
    all_res = np.empty((T, len(checkpoints)))

    for t in trange(T, desc=f"Batched b={b}"):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32)))
        all_res[t] = run_batched(m, n_max, b, rng, checkpoints, alloc_func)

    return all_res.mean(axis=0), all_res.std(axis=0)


def plot_all(checkpoints, results, m):
    plt.figure(figsize=(10, 6))
    for name, (mean, std) in results.items():
        plt.plot(checkpoints, mean, label=name)
        plt.fill_between(checkpoints, mean - std, mean + std, alpha=0.15)


    plt.axvline(m, color='k', linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(0.5, 40)
    plt.xlabel("n")
    plt.ylabel("Gap")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

def save_results_csv(checkpoints: np.ndarray, mean: np.ndarray, std: np.ndarray, filename: str):
    df = pd.DataFrame({'n': checkpoints, 'mean_gap': mean, 'std_gap': std})
    df.to_csv(filename, index=False)
    print(f"Saved results to {filename}")


if __name__ == "__main__":
    m = 100
    n_max = m * m
    T = 50
    seed = 42
    checkpoints = make_checkpoints(m, num_points=200)

    # 1) Sequential
    seq_results = {}
    beta_results = {}

    # one-choice
    mean_one, std_one = experiment_sequential(T, m, checkpoints, seed, one_choice)
    seq_results['one-choice (d=1)'] = (mean_one, std_one)
    save_results_csv(checkpoints, mean_one, std_one, f"one_choice_m{m}_T{T}.csv")

    # 1+beta for beta = 0.1, 0.3, 0.5, 0.7, 0.9
    mean_b01, std_b01 = experiment_sequential(T, m, checkpoints, seed + 5, lambda bins, rng: one_plus_beta_choice(bins, rng, beta=0.1))
    seq_results['(1+0.1)-choice'] = (mean_b01, std_b01)
    beta_results['(1+0.1)-choice'] = (mean_b01, std_b01)
    save_results_csv(checkpoints, mean_b01, std_b01, f"one_plus_beta0.1_m{m}_T{T}.csv")

    mean_b03, std_b03 = experiment_sequential(T, m, checkpoints, seed + 2, lambda bins, rng: one_plus_beta_choice(bins, rng, beta=0.3))
    seq_results['(1+0.3)-choice'] = (mean_b03, std_b03)
    beta_results['(1+0.3)-choice'] = (mean_b03, std_b03)
    save_results_csv(checkpoints, mean_b03, std_b03, f"one_plus_beta0.3_m{m}_T{T}.csv")

    mean_b05, std_b05 = experiment_sequential(T, m, checkpoints, seed + 3, lambda bins, rng: one_plus_beta_choice(bins, rng, beta=0.5))
    seq_results['(1+0.5)-choice'] = (mean_b05, std_b05)
    beta_results['(1+0.5)-choice'] = (mean_b05, std_b05)
    save_results_csv(checkpoints, mean_b05, std_b05, f"one_plus_beta0.5_m{m}_T{T}.csv")

    mean_b07, std_b07 = experiment_sequential(T, m, checkpoints, seed + 4, lambda bins, rng: one_plus_beta_choice(bins, rng, beta=0.7))
    seq_results['(1+0.7)-choice'] = (mean_b07, std_b07)
    beta_results['(1+0.7)-choice'] = (mean_b07, std_b07)
    save_results_csv(checkpoints, mean_b07, std_b07, f"one_plus_beta0.7_m{m}_T{T}.csv")

    mean_b09, std_b09 = experiment_sequential(T, m, checkpoints, seed + 6, lambda bins, rng: one_plus_beta_choice(bins, rng, beta=0.9))
    seq_results['(1+0.9)-choice'] = (mean_b09, std_b09)
    beta_results['(1+0.9)-choice'] = (mean_b09, std_b09)
    save_results_csv(checkpoints, mean_b09, std_b09, f"one_plus_beta0.9_m{m}_T{T}.csv")

    # two-choice
    mean_two, std_two = experiment_sequential(T, m, checkpoints, seed + 1, two_choice)
    seq_results['two-choice (d=2)'] = (mean_two, std_two)
    save_results_csv(checkpoints, mean_two, std_two, f"two_choice_m{m}_T{T}.csv")

    # three-choice
    mean_three, std_three = experiment_sequential(T, m, checkpoints, seed + 1, lambda bins, rng: d_choice(bins, rng, d=3))
    seq_results['three-choice (d=3)'] = (mean_three, std_three)
    save_results_csv(checkpoints, mean_three, std_three, f"three_choice_m{m}_T{T}.csv")

    # sequential comparison
    plot_all(checkpoints, seq_results, m)
    plot_all(checkpoints, beta_results, m)


    # same for batched arrivals

    batch_sizes = [m, 2 * m, 5 * m, 10 * m, 30 * m, 70 * m, 100 * m]

    for i, b in enumerate(batch_sizes):
        bat_results = {}
        # one-choice
        mean_one_b, std_one_b = experiment_batched(T, m, checkpoints, seed + 10 + i, one_choice, b)
        bat_results[f'one-choice (b={b})'] = (mean_one_b, std_one_b)
        save_results_csv(checkpoints, mean_one_b, std_one_b, f"one_choice_b{b}_m{m}_T{T}.csv")

        # 1 + beta=0.5
        mean_b05_b, std_b05_b = experiment_batched(T, m, checkpoints, seed + 40 + i,
                                                   lambda bins, rng: one_plus_beta_choice(bins, rng, 0.5), b)
        bat_results[f'(1+0.5)-choice (b={b})'] = (mean_b05_b, std_b05_b)
        save_results_csv(checkpoints, mean_b05_b, std_b05_b, f"one_plus_beta0.5_b{b}_m{m}_T{T}.csv")

        # two-choice (batched)
        mean_two_b, std_two_b = experiment_batched(T, m, checkpoints, seed + 20 + i, two_choice, b)
        bat_results[f'two-choice (b={b})'] = (mean_two_b, std_two_b)
        save_results_csv(checkpoints, mean_two_b, std_two_b, f"two_choice_b{b}_m{m}_T{T}.csv")

        plot_all(checkpoints, bat_results, m)

    # partial k1, partial k2,
    # plus one-, two- and 1+beta (beta=0.5) for reference
    partial_results = {}

    mean_pk1, std_pk1 = experiment_sequential(T, m, checkpoints, seed + 200, partial_k1)
    partial_results['partial k=1'] = (mean_pk1, std_pk1)
    save_results_csv(checkpoints, mean_pk1, std_pk1, f"partial_k1_m{m}_T{T}.csv")

    mean_pk2, std_pk2 = experiment_sequential(T, m, checkpoints, seed + 201, partial_k2)
    partial_results['partial k=2'] = (mean_pk2, std_pk2)
    save_results_csv(checkpoints, mean_pk2, std_pk2, f"partial_k2_m{m}_T{T}.csv")

    partial_results['one-choice (d=1)'] = (mean_one, std_one)
    partial_results['(1+0.5)-choice'] = (mean_b05, std_b05)
    partial_results['two-choice (d=2)'] = (mean_two, std_two)

    plot_all(checkpoints, partial_results, m)

    # partial-info with batching: for each b show partial k1, k2
    for i, b in enumerate(batch_sizes):
        partial_batched = {}

        mean_pk1_b, std_pk1_b = experiment_batched(T, m, checkpoints, seed + 300 + i, partial_k1, b)
        partial_batched[f'partial k=1 (b={b})'] = (mean_pk1_b, std_pk1_b)
        save_results_csv(checkpoints, mean_pk1_b, std_pk1_b, f"partial_k1_b{b}_m{m}_T{T}.csv")

        mean_pk2_b, std_pk2_b = experiment_batched(T, m, checkpoints, seed + 310 + i, partial_k2, b)
        partial_batched[f'partial k=2 (b={b})'] = (mean_pk2_b, std_pk2_b)
        save_results_csv(checkpoints, mean_pk2_b, std_pk2_b, f"partial_k2_b{b}_m{m}_T{T}.csv")

        # also include one/two/1+0.5 for reference
        mean_one_b_ref, std_one_b_ref = experiment_batched(T, m, checkpoints, seed + 320 + i, one_choice, b)
        partial_batched[f'one-choice (b={b})'] = (mean_one_b_ref, std_one_b_ref)


        mean_beta05_b_ref, std_beta05_b_ref = experiment_batched(T, m, checkpoints, seed + 340 + i,
                                                                 lambda bins, rng: one_plus_beta_choice(bins, rng, 0.5),
                                                                 b)
        partial_batched[f'(1+0.5)-choice (b={b})'] = (mean_beta05_b_ref, std_beta05_b_ref)

        mean_two_b_ref, std_two_b_ref = experiment_batched(T, m, checkpoints, seed + 330 + i, two_choice, b)
        partial_batched[f'two-choice (b={b})'] = (mean_two_b_ref, std_two_b_ref)

        plot_all(checkpoints, partial_batched, m)

