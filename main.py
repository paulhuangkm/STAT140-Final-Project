import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from tqdm import tqdm, trange
from scipy.stats import t, levene
from itertools import combinations

np.random.seed(42)
data = pd.read_csv('data.csv').drop([0, 1], axis=0)

assignment_s = data['Question strong'].fillna(0).to_numpy() != 0
assignment_w = data['Question weak'].fillna(0).to_numpy() != 0
assignment_n = data['Question no'].fillna(0).to_numpy() != 0
assignment_count = assignment_s.astype(int) + assignment_w.astype(int) + assignment_n.astype(int)

STRONG = 1
WEAK   = 2
NO_REL = 3
W_obs = (assignment_s * STRONG) + (assignment_w * WEAK) + (assignment_n * NO_REL)
pair_cols = ['Pair 1', 'Pair 2', 'Pair 3']

Y_pairs = data[pair_cols].replace({'1': 1, '2': 0}).astype(float)
Y_obs = Y_pairs.mean(axis=1).to_numpy()
valid_y = Y_pairs.notna().all(axis=1)

print(assignment_count)
valid_assignment = assignment_count == 1
valid = valid_assignment & valid_y

data = data[valid].copy()
W_obs = W_obs[valid]
Y_obs = Y_obs[valid]

print(W_obs)
print(Y_obs)
print('=' * 10 + ' STRONG ' + '=' * 10)
print(Y_obs[W_obs == STRONG])
print('mean =', Y_obs[W_obs == STRONG].mean())
print('std  =', Y_obs[W_obs == STRONG].std(ddof=1))
print('=' * 10 + ' WEAK ' + '=' * 10)
print(Y_obs[W_obs == WEAK])
print('mean =', Y_obs[W_obs == WEAK].mean())
print('std  =', Y_obs[W_obs == WEAK].std(ddof=1))
print('=' * 10 + ' NOT RELATED ' + '=' * 10)
print(Y_obs[W_obs == NO_REL])
print('mean =', Y_obs[W_obs == NO_REL].mean())
print('std  =', Y_obs[W_obs == NO_REL].std(ddof=1))

N  = len(W_obs)
Ns = [(W_obs == STRONG).sum(), (W_obs == WEAK).sum(), (W_obs == NO_REL).sum()]
print(f'{N = }')
print(f'N_level = {Ns}')
J  = 3

def calc_F(Y, W):
    N = len(Y)
    J = 3
    mean  = Y.mean()
    means = np.array([Y[W == STRONG].mean(), Y[W == WEAK].mean(), Y[W == NO_REL].mean()])
    Njs   = np.array([(W == STRONG).sum(), (W == WEAK).sum(), (W == NO_REL).sum()])
    MS_treatment = (Njs * ((means - mean) ** 2)).sum() / (J - 1)
    MS_residual  = ((Y - means[W - 1]) ** 2).sum() / (N - J)
    return MS_treatment / MS_residual

##### ANOVA F-statistic for approx. p-value
F_obs         = calc_F(Y_obs, W_obs)
N_rand        = 100_000

F_rands = []
for i in trange(N_rand):
    W = W_obs[np.random.permutation(N)]
    F_rands.append(calc_F(Y_obs, W))

F_rands = np.stack(F_rands)
print()
print(f'{F_obs   = :.05f}')
print(f'p-value = {(F_rands >= F_obs).mean():.05f}')

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(8, 6))
plt.hist(F_rands, bins=100, label='Randomized F statistics')
plt.axvline(x=F_obs, color='r', linewidth=3, label='Observed F statistic')
plt.xlabel("F statistic")
plt.ylabel("Count")
plt.title("Null Randomization Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("null.png")

##### Pair-wise average causal effect
# Fisher-exact p-value
# Neymanian CI

def calc_t(Y, W):
    return (Y[W == 1].mean() - Y[W == 0].mean())

def fisher_exact(Y, W, alpha=0.05):
    n  = len(Y)
    n1 = (W == 1).sum()
    print(f'{n  = }')
    print(f'{n1 = }')

    assignments = np.zeros((math.comb(n, n1), n), dtype=np.int32)

    for i, indices in enumerate(combinations(range(n), n1)):
        for index in indices:
            assignments[i][index] = 1

    T_obs = calc_t(Y, W)

    print()
    print('T_obs =', T_obs)
    Ts = []
    for assignment in assignments:
        Ts.append(calc_t(Y, assignment))
    eps = 1e-12
    print(f'p-value = {(np.abs(Ts) >= abs(T_obs) - eps).mean()}')

    s1_sq_N1 = Y[W == 1].var(ddof=1) / n1
    s0_sq_N0 = Y[W == 0].var(ddof=1) / (n - n1)

    var_hat = s1_sq_N1 + s0_sq_N0
    t_df    = ((s1_sq_N1 + s0_sq_N0) ** 2) / ((s0_sq_N0 ** 2) / (n-n1-1) + (s1_sq_N1 ** 2) / (n1-1))

    # asymp_p_val = 1 - t.cdf(T_obs / np.sqrt(var_hat), df=t_df)
    test_stat = T_obs / np.sqrt(var_hat)
    asymp_p_val = 2 * (1 - t.cdf(abs(test_stat), df=t_df))

    print(f'Asymptotic p-value = {asymp_p_val}')

    # Neyman/Welch-style confidence interval
    t_crit = t.ppf(1 - alpha / 2, df=t_df)
    ci_low = T_obs - t_crit * np.sqrt(var_hat)
    ci_high = T_obs + t_crit * np.sqrt(var_hat)

    print(f'{100 * (1-alpha):.1f}% CI = ({ci_low}, {ci_high})')

    return {
        "T_obs": T_obs,
        "randomization_p_value": (np.abs(Ts) >= abs(T_obs) - eps).mean(),
        "asymptotic_p_value": asymp_p_val,
        "var_hat": var_hat,
        "t_df": t_df,
        "ci": (ci_low, ci_high),
    }

group_1or2 = (W_obs == 1) | (W_obs == 2)
group_1or3 = (W_obs == 1) | (W_obs == 3)
group_2or3 = (W_obs == 2) | (W_obs == 3)

# strong vs weak (strong = 1)
print('=' * 20)
print(f'STRONG v.s. WEAK')
fisher_exact(Y_obs[group_1or2], W_obs[group_1or2] == STRONG)

# strong vs no (strong = 1)
print('=' * 20)
print(f'STRONG v.s. NOT RELATED')
fisher_exact(Y_obs[group_1or3], W_obs[group_1or3] == STRONG)

# weak vs no (weak = 1)
print('=' * 20)
print(f'WEAK v.s. NOT RELATED')
fisher_exact(Y_obs[group_2or3], W_obs[group_2or3] == WEAK)

##### Regression (X = indicator of 3 levels of treatments \in R^3)
# X = 
def fit_regression_sm(Y, W, alpha=0.05):
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W)

    X = pd.DataFrame({
        "STRONG": (W == STRONG).astype(float),
        "WEAK": (W == WEAK).astype(float),
    })
    X = sm.add_constant(X)

    model = sm.OLS(Y, X)
    res = model.fit(cov_type="HC3")

    ci = res.conf_int(alpha=alpha)

    summary = pd.DataFrame({
        "coef": res.params,
        "robust_se": res.bse,
        "t": res.tvalues,
        "p_value": res.pvalues,
        "ci_low": ci.iloc[:, 0],
        "ci_high": ci.iloc[:, 1],
    })

    print(summary)
    return res, summary

def regression_F_stat_sm(Y, W):
    """
    Returns the model F-statistic from OLS (with intercept, STRONG, WEAK).
    """
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W)

    X = np.column_stack([
        (W == STRONG).astype(float),
        (W == WEAK).astype(float),
    ])
    X = sm.add_constant(X)

    res = sm.OLS(Y, X).fit()  # classical fit for F-stat
    return res.fvalue

def fisher_regression_test_sm(Y, W, N_rand=100000, seed=42):
    """
    Randomization test using the regression F-statistic.
    """
    rng = np.random.default_rng(seed)

    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W)

    F_obs = regression_F_stat_sm(Y, W)

    F_rands = []
    for _ in range(N_rand):
        W_perm = rng.permutation(W)
        F_rands.append(regression_F_stat_sm(Y, W_perm))

    F_rands = np.array(F_rands)
    p_value = (F_rands >= F_obs).mean()

    print(f"Observed F-stat = {F_obs}")
    print(f"Randomization p-value = {p_value}")

    return {
        "F_obs": F_obs,
        "p_value": p_value,
        "F_rands": F_rands
    }
res, reg_summary = fit_regression_sm(Y_obs, W_obs)
fisher_regression_test_sm(Y_obs, W_obs, N_rand=100000)


groups = {
    "STRONG": Y_obs[W_obs == STRONG],
    "WEAK": Y_obs[W_obs == WEAK],
    "NO_REL": Y_obs[W_obs == NO_REL],
}

stat, p = levene(groups["STRONG"], groups["WEAK"], groups["NO_REL"], center="median")
print("Brown-Forsythe statistic:", stat)
print("Brown-Forsythe p-value:", p)

var_strong = groups["STRONG"].var(ddof=1)
var_weak = groups["WEAK"].var(ddof=1)
var_no = groups["NO_REL"].var(ddof=1)

print("Variance ratio STRONG / WEAK:", var_strong / var_weak)
print("Variance ratio STRONG / NO_REL:", var_strong / var_no)

plt.figure(figsize=(7, 5))
plt.boxplot(
    [groups["STRONG"], groups["WEAK"], groups["NO_REL"]],
    labels=["Strong", "Weak", "Neutral"]
)
plt.ylabel("Observed Outcome")
plt.xlabel("Treatment Group")
plt.title("Box-plot of Observed Outcome")
plt.tight_layout()
plt.savefig("variance_boxplot.png")