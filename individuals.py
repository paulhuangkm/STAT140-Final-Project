import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import levene
from tqdm import trange

np.random.seed(42)

STRONG = 1
WEAK   = 2
NO_REL = 3

pair_cols = ["Pair 1", "Pair 2", "Pair 3"]

data = pd.read_csv("data.csv").drop([0, 1], axis=0)

assignment_s = data["Question strong"].fillna(0).to_numpy() != 0
assignment_w = data["Question weak"].fillna(0).to_numpy() != 0
assignment_n = data["Question no"].fillna(0).to_numpy() != 0
assignment_count = assignment_s.astype(int) + assignment_w.astype(int) + assignment_n.astype(int)

W_obs = (
    assignment_s.astype(int) * STRONG
    + assignment_w.astype(int) * WEAK
    + assignment_n.astype(int) * NO_REL
)

Y_pairs = data[pair_cols].replace({"1": 1, "2": 0}).astype(float)

valid_assignment = assignment_count == 1
valid_y = Y_pairs.notna().all(axis=1)
valid = valid_assignment & valid_y

data = data[valid].copy()
W_obs = W_obs[valid]
Y_pairs = Y_pairs[valid].copy()

def calc_F(Y, W):
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W)

    N = len(Y)
    J = 3

    overall_mean = Y.mean()

    means = np.array([
        Y[W == STRONG].mean(),
        Y[W == WEAK].mean(),
        Y[W == NO_REL].mean(),
    ])

    Njs = np.array([
        (W == STRONG).sum(),
        (W == WEAK).sum(),
        (W == NO_REL).sum(),
    ])

    MS_treatment = (Njs * (means - overall_mean) ** 2).sum() / (J - 1)

    residuals = Y - means[W - 1]
    MS_residual = (residuals ** 2).sum() / (N - J)

    # Handle pair-level binary outcomes where within-group variance can be zero
    if np.isclose(MS_residual, 0):
        if np.isclose(MS_treatment, 0):
            return 0.0
        return np.inf

    return MS_treatment / MS_residual

def randomization_F_test(Y, W, N_rand=100_000):
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W)

    F_obs = calc_F(Y, W)

    F_rands = []
    for _ in trange(N_rand, leave=False):
        W_perm = np.random.permutation(W)
        F_rands.append(calc_F(Y, W_perm))

    F_rands = np.array(F_rands)

    if np.isinf(F_obs):
        p_value = np.isinf(F_rands).mean()
    else:
        p_value = (F_rands >= F_obs).mean()

    return F_obs, p_value

def fit_regression_sm(Y, W, alpha=0.05):
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W)

    X = pd.DataFrame({
        "STRONG": (W == STRONG).astype(float),
        "WEAK": (W == WEAK).astype(float),
    })
    X = sm.add_constant(X)

    res = sm.OLS(Y, X).fit(cov_type="HC3")
    ci = res.conf_int(alpha=alpha)

    summary = pd.DataFrame({
        "coef": res.params,
        "robust_se": res.bse,
        "t": res.tvalues,
        "p_value": res.pvalues,
        "ci_low": ci.iloc[:, 0],
        "ci_high": ci.iloc[:, 1],
    })

    return summary

def analyze_one_pair(pair_col):
    Y = Y_pairs[pair_col].to_numpy()

    print("\n" + "=" * 60)
    print(f"Analysis for {pair_col}")
    print("=" * 60)

    groups = {
        "STRONG": Y[W_obs == STRONG],
        "WEAK": Y[W_obs == WEAK],
        "NO_REL": Y[W_obs == NO_REL],
    }

    print("\nDescriptive statistics")
    for name, vals in groups.items():
        print(
            f"{name:8s} "
            f"n={len(vals):2d}, "
            f"mean={vals.mean():.4f}, "
            f"std={vals.std(ddof=1):.4f}, "
            f"var={vals.var(ddof=1):.4f}"
        )

    print("\nRandomization ANOVA/F-test")
    F_obs, p_rand = randomization_F_test(Y, W_obs)

    if np.isinf(F_obs):
        print("F_obs = inf")
    else:
        print(f"F_obs = {F_obs:.5f}")

    print(f"randomization p-value = {p_rand:.5f}")

    print("\nBrown-Forsythe variance test")
    bf_stat, bf_p = levene(
        groups["STRONG"],
        groups["WEAK"],
        groups["NO_REL"],
        center="median",
    )
    print(f"Brown-Forsythe statistic = {bf_stat:.5f}")
    print(f"Brown-Forsythe p-value = {bf_p:.5f}")

    print("\nVariance ratios")
    var_s = groups["STRONG"].var(ddof=1)
    var_w = groups["WEAK"].var(ddof=1)
    var_n = groups["NO_REL"].var(ddof=1)

    print(f"STRONG / WEAK   = {var_s / var_w if var_w > 0 else np.nan:.5f}")
    print(f"STRONG / NO_REL = {var_s / var_n if var_n > 0 else np.nan:.5f}")
    print(f"WEAK / NO_REL   = {var_w / var_n if var_n > 0 else np.nan:.5f}")

    print("\nOLS regression with HC3 robust SEs")
    reg_summary = fit_regression_sm(Y, W_obs)
    print(reg_summary)

    return {
        "pair": pair_col,
        "F_obs": F_obs,
        "anova_randomization_p": p_rand,
        "brown_forsythe_stat": bf_stat,
        "brown_forsythe_p": bf_p,
        "regression_summary": reg_summary,
    }

all_results = []

for pair_col in pair_cols:
    result = analyze_one_pair(pair_col)
    all_results.append(result)