# SCENARIO 1 (BROAD: 100K - 1M)

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import msprime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score  

def generate_sfs(ne, n_replicates=5000, n_diploids=9, seq_len=10_000,
                 mut_rate=1.25e-8, rec_rate=1e-8, seed=42):
    sfs_vectors = []
    rng = np.random.default_rng(seed)
    for _ in tqdm(range(n_replicates), desc=f"Simulating Ne={ne:,}"):
        ts = msprime.sim_ancestry(
            samples=2 * n_diploids,
            recombination_rate=rec_rate,
            sequence_length=seq_len,
            population_size=ne,
            random_seed=int(rng.integers(1, 1_000_000_000)),
        )
        ts_mut = msprime.sim_mutations(
            ts, rate=mut_rate, random_seed=int(rng.integers(1, 1_000_000_000))
        )
        sfs = ts_mut.allele_frequency_spectrum(
            mode="site", polarised=False, span_normalise=False
        )
        sfs_folded = sfs[1:10]  # drop monomorphic bin 0
        if sfs_folded.sum() == 0:
            continue
        sfs_vectors.append(sfs_folded / sfs_folded.sum())
    return np.array(sfs_vectors), None


# Class bins used in SVM
ne_values = [100_000, 500_000, 1_000_000]  # class bins
X = []  # SFS vectors
y = []  # True Ne values

for ne in ne_values:
    sfs_vectors, _ = generate_sfs(ne=ne, n_replicates=5000)
    X.extend(sfs_vectors)
    y.extend([ne] * len(sfs_vectors))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train RF regressor
reg = RandomForestRegressor()
reg.fit(X_train, y_train)

# Predict
y_pred = reg.predict(X_test)

# Regression metrics
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# --- Convert predictions to nearest class bin for accuracy ---
def bin_prediction(pred, bins):
    """Map continuous prediction to nearest class bin."""
    return min(bins, key=lambda x: abs(x - pred))

# Apply binning
y_pred_binned = [bin_prediction(pred, ne_values) for pred in y_pred]
y_test_binned = [bin_prediction(true, ne_values) for true in y_test]  

# Accuracy
rf_accuracy = accuracy_score(y_test_binned, y_pred_binned)
print("RF Classification Accuracy (via binning):", rf_accuracy)


# SCENARIO 2 (NARROW: 50K - 500K)

# Class bins you used in SVM
ne_values = [50_000, 225_000, 500_000]  # class bins
X = []  # SFS vectors
y = []  # True Ne values

for ne in ne_values:
    sfs_vectors, _ = generate_sfs(ne=ne, n_replicates=5000)
    X.extend(sfs_vectors)
    y.extend([ne] * len(sfs_vectors))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train RF regressor
reg = RandomForestRegressor()
reg.fit(X_train, y_train)

# Predict
y_pred = reg.predict(X_test)

# Regression metrics
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# --- Convert predictions to nearest class bin for accuracy ---
def bin_prediction(pred, bins):
    """Map continuous prediction to nearest class bin."""
    return min(bins, key=lambda x: abs(x - pred))

# Apply binning
y_pred_binned = [bin_prediction(pred, ne_values) for pred in y_pred]
y_test_binned = [bin_prediction(true, ne_values) for true in y_test]  

# Accuracy
rf_accuracy = accuracy_score(y_test_binned, y_pred_binned)
print("RF Classification Accuracy (via binning):", rf_accuracy)
