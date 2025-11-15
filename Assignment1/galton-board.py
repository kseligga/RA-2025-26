import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import time

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def binomial_pmf(n):
    pmf = np.array([math.comb(n, k) * (0.5 ** n) for k in range(n + 1)], dtype=float)
    return pmf


def normal_pdf_at_integers(n):
    mu = n / 2.0
    sigma = math.sqrt(n) / 2.0  # stddev
    ks = np.arange(0, n + 1)
    pdf = (1.0 / (math.sqrt(2 * math.pi) * sigma)) * np.exp(-0.5 * ((ks - mu) / sigma) ** 2)
    return pdf


def simulate_galton_vectorized(n, N, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
        samples = rng.binomial(n, 0.5, size=N)
    else:
        samples = np.random.binomial(n, 0.5, size=N)
    counts = np.bincount(samples, minlength=n + 1)
    return counts


def compute_errors(n, counts, return_dict=False):

    N = int(np.sum(counts))
    emp_freq = counts / N
    pmf = binomial_pmf(n)
    normal_pdf = normal_pdf_at_integers(n)
    mse_emp_vs_binom = np.mean((emp_freq - pmf) ** 2)
    mse_emp_vs_normal = np.mean((emp_freq - normal_pdf) ** 2)
    mse_binom_vs_normal = np.mean((pmf - normal_pdf) ** 2)
    if return_dict:
        return {
            "n": n,
            "N": N,
            "mse_emp_vs_binom": mse_emp_vs_binom,
            "mse_emp_vs_normal": mse_emp_vs_normal,
            "mse_binom_vs_normal": mse_binom_vs_normal,
            "emp_freq": emp_freq,
            "pmf": pmf,
            "normal_pdf": normal_pdf,
        }
    else:
        return emp_freq, pmf, normal_pdf, mse_emp_vs_binom, mse_emp_vs_normal, mse_binom_vs_normal


def plot_empirical_vs_theory(n, counts, title_suffix=None, save_path=None):
    """a plot with empirical histogram, binomial pmf line, and normal pdf line"""
    ks = np.arange(0, n + 1)
    N = np.sum(counts)
    emp_freq = counts / N
    pmf = binomial_pmf(n)
    normal_pdf = normal_pdf_at_integers(n)

    plt.figure(figsize=(9, 5))
    # empirical bar
    plt.bar(ks, emp_freq, label="Empirical freq (counts/N)", alpha=0.7)
    # theoretical pmf
    plt.plot(ks, pmf, marker="o", linestyle="-", label="Binomial pmf Bin(n, 1/2)")
    # normal pdf as smooth line
    plt.plot(ks, normal_pdf, linestyle="-", label=f"Normal pdf N(n/2, n/4) at integers")
    plt.xlabel("k (number of rights)")
    plt.ylabel("Probability / Frequency")
    title = f"Galton board simulation n={n}, N={N}"
    if title_suffix:
        title += " - " + title_suffix
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def study_error_vs_N(n, N_list, seed=42):

    results = []
    for i, N in enumerate(N_list):
        s = seed + i * 1000
        counts = simulate_galton_vectorized(n, N, seed=s)
        res = compute_errors(n, counts, return_dict=True)
        results.append({
            "N": N,
            "mse_emp_vs_binom": res["mse_emp_vs_binom"],
            "mse_emp_vs_normal": res["mse_emp_vs_normal"],
            "mse_binom_vs_normal": res["mse_binom_vs_normal"],
        })
        plot_empirical_vs_theory(n, counts, title_suffix=f"MSE_emp_vs_binom={res['mse_emp_vs_binom']:.3e}")
    df = pd.DataFrame(results)
    return df


def study_error_vs_n(n_list, N, seed=42):

    results = []
    for i, n in enumerate(n_list):
        s = seed + i * 1000
        counts = simulate_galton_vectorized(n, N, seed=s)
        res = compute_errors(n, counts, return_dict=True)
        results.append({
            "n": n,
            "mse_emp_vs_binom": res["mse_emp_vs_binom"],
            "mse_emp_vs_normal": res["mse_emp_vs_normal"],
            "mse_binom_vs_normal": res["mse_binom_vs_normal"],
        })
        plot_empirical_vs_theory(n, counts, title_suffix=f"MSE_emp_vs_binom={res['mse_emp_vs_binom']:.3e}")
    df = pd.DataFrame(results)
    return df


#### USAGE ####
n = 20
N = 5000
seed = 42

# Study on N increases (for fixed n)
N_list = [100, 500, 1000, 2000, 5000, 10000]
df_N = study_error_vs_N(n, N_list, seed=seed)
print(df_N)

plt.figure(figsize=(8, 4))
plt.errorbar(df_N["N"], df_N["mse_emp_vs_binom"], marker="o", linestyle="-", label="Emp vs Bin")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("N (log scale)")
plt.ylabel("Mean squared error (log scale)")
plt.title(f"Error behavior vs N (n={n})")
plt.legend()
plt.tight_layout()
plt.show()


# Study on n increases (for fixed N)
n_list = [4, 8, 12, 20, 40, 80]
df_n = study_error_vs_n(n_list, N=2000, seed=seed)
print(df_n)

plt.figure(figsize=(8, 4))
plt.plot(df_n["n"], df_n["mse_emp_vs_binom"], marker="o", linestyle="-", label="Emp vs Bin")
plt.plot(df_n["n"], df_n["mse_binom_vs_normal"], marker="o", linestyle="-", label="Bin vs Normal (theoretical)")
plt.yscale("log")
plt.xlabel("n (levels)")
plt.ylabel("Mean squared error (pmf vs pdf at integers, log scale)")
plt.title("Theoretical Binomial vs Normal MSE as n increases")
plt.legend()
plt.tight_layout()
plt.show()
