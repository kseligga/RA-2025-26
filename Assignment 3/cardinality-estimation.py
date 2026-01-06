import numpy as np
import mmh3
import math
import os
import glob
import csv
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_zipf_stream(n, alpha, N):
    if alpha == 0:
        probs = np.ones(n) / n
    else:
        i = np.arange(1, n + 1)
        weights = 1 / np.power(i, alpha)
        c_n = 1 / np.sum(weights)
        probs = c_n * weights
    return np.random.choice(np.arange(1, n + 1), size=N, p=probs)


def get_hash(item, seed=0):
    val = str(item).encode('utf-8')
    return mmh3.hash(val, seed=seed, signed=False)


class HyperLogLog:
    def __init__(self, b=10, seed=0):
        self.b = b
        self.m = 1 << b
        self.M = np.zeros(self.m, dtype=int)
        self.seed = seed
        if self.m == 16:
            self.alpha_m = 0.673
        elif self.m == 32:
            self.alpha_m = 0.697
        elif self.m == 64:
            self.alpha_m = 0.709
        else:
            self.alpha_m = 0.7213 / (1 + 1.079 / self.m)

    def add(self, item):
        x = get_hash(item, self.seed)
        j = x & (self.m - 1)
        w = x >> self.b
        rank = 1
        while (w & 1) == 0 and rank <= (32 - self.b):
            w >>= 1
            rank += 1
        self.M[j] = max(self.M[j], rank)

    def count(self):
        Z = 1.0 / np.sum(2.0 ** -self.M)
        E = self.alpha_m * (self.m ** 2) * Z
        if E <= 2.5 * self.m:
            V = np.count_nonzero(self.M == 0)
            if V > 0: E = self.m * math.log(self.m / V)
        return E


class Recordinality:
    def __init__(self, k=32, seed=0):
        self.k = k
        self.seed = seed
        self.S = set()
        self.R = 0

    def add(self, item):
        h_val = get_hash(item, self.seed)
        entry = (h_val, item)
        if entry in self.S: return
        if len(self.S) < self.k:
            self.S.add(entry)
            self.R += 1
        else:
            min_entry = min(self.S, key=lambda x: x[0])
            if h_val > min_entry[0]:
                self.S.remove(min_entry)
                self.S.add(entry)
                self.R += 1

    def count(self):
        if len(self.S) < self.k: return len(self.S)
        return self.k * ((1.0 + 1.0 / self.k) ** (self.R - self.k + 1)) - 1


class KMV:
    def __init__(self, k=32, seed=0):
        self.k = k
        self.seed = seed
        self.max_hash_val = 2 ** 32 - 1
        self.buffer = set()

    def add(self, item):
        h = get_hash(item, self.seed)
        if len(self.buffer) < self.k:
            self.buffer.add(h)
        else:
            current_max = max(self.buffer)
            if h < current_max and h not in self.buffer:
                self.buffer.remove(current_max)
                self.buffer.add(h)

    def count(self):
        if len(self.buffer) < self.k: return len(self.buffer)
        return (self.k - 1) / (max(self.buffer) / self.max_hash_val)


def process_file_content(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            for word in line.split():
                yield word


def get_true_cardinality_from_dat(dat_path):
    with open(dat_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


if __name__ == "__main__":
    base_path = r"C:\Users\kubas\Downloads\datasetsRA\datasets"
    txt_files = glob.glob(os.path.join(base_path, "*.txt"))
    csv_file = "results_summary.csv"
    hll_b_vals = [4, 6, 8, 10, 12]
    rec_k_vals = [16, 64, 256, 512, 1024]
    trials = 5
    dracula_emp_hll = []

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Algorithm", "Param_Value", "True_Cardinality", "Est_Mean", "Rel_Error"])

        for txt_file in tqdm(txt_files):
            filename = os.path.basename(txt_file)
            dat_file = txt_file.replace('.txt', '.dat')
            true_card = get_true_cardinality_from_dat(dat_file)
            data = list(process_file_content(txt_file))
            is_dracula = "dracula" in filename.lower()

            for b in hll_b_vals if is_dracula else [10]:
                ests = []
                for t in range(trials):
                    algo = HyperLogLog(b=b, seed=t)
                    for item in data: algo.add(item)
                    ests.append(algo.count())
                mean_est = np.mean(ests)
                rel_err = abs(mean_est - true_card) / true_card
                writer.writerow([filename, "HLL", f"b={b}", true_card, mean_est, rel_err])
                if is_dracula: dracula_emp_hll.append(rel_err)

            for k in (rec_k_vals if is_dracula else [256]):
                for a_name, a_class in [("REC", Recordinality), ("KMV", KMV)]:
                    ests = []
                    for t in range(trials):
                        inst = a_class(k=k, seed=t)
                        for item in data: inst.add(item)
                        ests.append(inst.count())
                    mean_e = np.mean(ests)
                    writer.writerow(
                        [filename, a_name, f"k={k}", true_card, mean_e, abs(mean_e - true_card) / true_card])

        n_distinct, stream_len = 5000, 50000
        for alpha in [0.0, 1.0, 2.0]:
            stream = generate_zipf_stream(n_distinct, alpha, stream_len)
            true_card = len(np.unique(stream))
            for a_name, a_class, p in [("HLL", HyperLogLog, 10), ("REC", Recordinality, 256)]:
                ests = [a_class(p, seed=t) for t in range(trials)]
                for algo in ests:
                    for item in stream: algo.add(item)
                mean_est = np.mean([a.count() for a in ests])
                writer.writerow(
                    [f"Synthetic_a{alpha}", a_name, p, true_card, mean_est, abs(mean_est - true_card) / true_card])

    if dracula_emp_hll:
        m_vals = [2 ** b for b in hll_b_vals]
        theory = [1.04 / np.sqrt(m) for m in m_vals]
        plt.figure(figsize=(10, 5))
        plt.plot(m_vals, theory, 'r--', label='Theory (1.04/sqrt(m))')
        plt.plot(m_vals, dracula_emp_hll, 'bo-', label='Empirical (Dracula)')
        plt.xscale('log', base=2)
        plt.yscale('log')
        plt.xlabel('m (counters)')
        plt.ylabel('Rel Error')
        plt.title('HLL: Theory vs Empirical (Dracula)')
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.show()