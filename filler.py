import numpy as np
import pandas as pd

from scipy.stats import t, levene

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

dummy_cols = ['Pair 4', 'Pair 5', 'Pair 6', 'Pair 7', 'Pair 8', 'Pair 9', 'Pair 10']

Y_dummy_pairs = data[dummy_cols].replace({'1': 1, '2': 0}).astype(float)
valid_dummy = Y_dummy_pairs.notna().all(axis=1)

Y_dummy = Y_dummy_pairs[valid_dummy].mean(axis=1).to_numpy()
W_dummy = W_obs[valid_dummy]

groups_dummy = {
    "STRONG": Y_dummy[W_dummy == STRONG],
    "WEAK": Y_dummy[W_dummy == WEAK],
    "NO_REL": Y_dummy[W_dummy == NO_REL],
}

print("Dummy pair summaries")
for name, vals in groups_dummy.items():
    print(
        name,
        "n =", len(vals),
        "mean =", vals.mean(),
        "std =", vals.std(ddof=1),
        "var =", vals.var(ddof=1),
    )

from scipy.stats import levene

stat, p = levene(
    groups_dummy["STRONG"],
    groups_dummy["WEAK"],
    groups_dummy["NO_REL"],
    center="median",
)

print("Dummy Brown-Forsythe statistic:", stat)
print("Dummy Brown-Forsythe p-value:", p)