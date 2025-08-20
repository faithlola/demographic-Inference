#SCENARIO 1 (BROAD: 100K - 1M)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import msprime

# --- 1. Generate SFS data with class labels ---
def generate_sfs(ne, n_replicates=5000, n_diploids=9, seq_len=10000, mut_rate=1.25e-8):
    sfs_vectors = []
    snp_counts = []
    np.random.seed(42)

    for _ in tqdm(range(n_replicates), desc=f"Simulating Ne={ne}"):
        ts = msprime.sim_ancestry(
            samples=2 * n_diploids,
            recombination_rate=1e-8,
            sequence_length=seq_len,
            population_size=ne,
            random_seed=np.random.randint(1, 1e9)
        )
        ts_mut = msprime.sim_mutations(
            ts, rate=mut_rate, random_seed=np.random.randint(1, 1e9)
        )
        snp_counts.append(ts_mut.num_sites)
        sfs = ts_mut.allele_frequency_spectrum(
            mode="site", polarised=False, span_normalise=False
        )
        sfs_folded = sfs[1:10]
        if sfs_folded.sum() == 0:
            continue
        sfs_vectors.append(sfs_folded / sfs_folded.sum())
    return np.array(sfs_vectors), snp_counts

# Define Ne bins and class labels
ne_values = [100_000, 500_000, 1_000_000]
labels = list(range(len(ne_values)))  # [0, 1, 2]

# Simulate data
sfs_all = []
y_all = []
for i, ne in enumerate(ne_values):
    sfs_vectors, _ = generate_sfs(ne=ne, n_replicates=5000)
    sfs_all.append(sfs_vectors)
    y_all.extend([i] * len(sfs_vectors))

X = np.vstack(sfs_all)
y = np.array(y_all)

# --- 2. Split train/test once ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- 3. Train SVM Classifier ---
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

# --- 4. Train RF Classifier ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# --- 5. Permutation Test Function ---
def paired_permutation_test(y_true, y_pred_svm, y_pred_rf, n_permutations=10000):
    svm_acc = accuracy_score(y_true, y_pred_svm)
    rf_acc = accuracy_score(y_true, y_pred_rf)
    observed_diff = rf_acc - svm_acc

    diffs = []
    for _ in range(n_permutations):
        swapped_rf, swapped_svm = [], []
        for i in range(len(y_true)):
            if np.random.rand() > 0.5:
                swapped_rf.append(y_pred_rf[i])
                swapped_svm.append(y_pred_svm[i])
            else:
                swapped_rf.append(y_pred_svm[i])
                swapped_svm.append(y_pred_rf[i])
        new_rf_acc = accuracy_score(y_true, swapped_rf)
        new_svm_acc = accuracy_score(y_true, swapped_svm)
        diffs.append(new_rf_acc - new_svm_acc)

    diffs = np.array(diffs)
    p_value = np.mean(diffs >= observed_diff)
    return observed_diff, diffs, p_value, svm_acc, rf_acc

# --- 6. Run the permutation test ---
observed_diff, diff_distribution, p_val, svm_acc, rf_acc = paired_permutation_test(
    y_true=y_test,
    y_pred_svm=svm_preds,
    y_pred_rf=rf_preds,
    n_permutations=10000
)

# --- 7. Print and plot results ---
print(f"\nSVM Accuracy: {svm_acc:.2f}")
print(f"RF Accuracy: {rf_acc:.2f}")
print(f"Observed Accuracy Difference (RF - SVM): {observed_diff:.3f}")
print(f"P-value: {p_val:.4f}")

plt.hist(diff_distribution, bins=30, color='slateblue', edgecolor='black')
plt.axvline(observed_diff, color='red', linestyle='dashed', linewidth=2)
plt.title("Permutation Test: Accuracy Difference (RF - SVM (100,000 - 1,000,000))")
plt.xlabel("Accuracy Difference")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Optional: Display confusion matrix
cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{ne//1000}k" for ne in ne_values])
disp.plot()
plt.title("RF Confusion Matrix")
plt.show()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# For SVM
cm_svm = confusion_matrix(y_test, svm_preds)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=["100k", "500k", "1m"])
disp_svm.plot(cmap="plasma")  # Optional: pick your color map
plt.title("SVM Confusion Matrix")
plt.show()

#SCENARIO 2 NARROW (50K - 500K)


# --- 1. Generate SFS ---
def generate_sfs(ne, n_replicates=5000, n_diploids=9, seq_len=10000, mut_rate=1.25e-8):
    sfs_vectors = []
    snp_counts = []
    np.random.seed(42)

    for _ in tqdm(range(n_replicates), desc=f"Simulating Ne={ne}"):
        ts = msprime.sim_ancestry(
            samples=2 * n_diploids,
            recombination_rate=1e-8,
            sequence_length=seq_len,
            population_size=ne,
            random_seed=np.random.randint(1, 1e9)
        )
        ts_mut = msprime.sim_mutations(
            ts, rate=mut_rate, random_seed=np.random.randint(1, 1e9)
        )
        snp_counts.append(ts_mut.num_sites)

        sfs = ts_mut.allele_frequency_spectrum(
            mode="site", polarised=False, span_normalise=False
        )
        sfs_folded = sfs[1:10]
        if sfs_folded.sum() == 0:
            continue
        sfs_vectors.append(sfs_folded / sfs_folded.sum())

    return np.array(sfs_vectors), snp_counts

# --- 2. Define Ne bins and simulate data ---
ne_values = [50_000, 225_000, 500_000]
labels = list(range(len(ne_values)))  # [0, 1, 2]

sfs_all = []
y_all = []
for i, ne in enumerate(ne_values):
    sfs_vectors, _ = generate_sfs(ne=ne, n_replicates=5000)
    sfs_all.append(sfs_vectors)
    y_all.extend([i] * len(sfs_vectors))  # Use class labels here

X = np.vstack(sfs_all)
y = np.array(y_all)

# --- 3. Split train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- 4. Train models ---
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# --- 5. Permutation test ---
def paired_permutation_test(y_true, y_pred_svm, y_pred_rf, n_permutations=10000):
    svm_acc = accuracy_score(y_true, y_pred_svm)
    rf_acc = accuracy_score(y_true, y_pred_rf)
    observed_diff = rf_acc - svm_acc

    diffs = []
    for _ in range(n_permutations):
        swapped_rf, swapped_svm = [], []
        for i in range(len(y_true)):
            if np.random.rand() > 0.5:
                swapped_rf.append(y_pred_rf[i])
                swapped_svm.append(y_pred_svm[i])
            else:
                swapped_rf.append(y_pred_svm[i])
                swapped_svm.append(y_pred_rf[i])
        new_rf_acc = accuracy_score(y_true, swapped_rf)
        new_svm_acc = accuracy_score(y_true, swapped_svm)
        diffs.append(new_rf_acc - new_svm_acc)

    diffs = np.array(diffs)
    p_value = np.mean(diffs >= observed_diff)
    return observed_diff, diffs, p_value, svm_acc, rf_acc

# --- 6. Run test ---
observed_diff, diff_distribution, p_val, svm_acc, rf_acc = paired_permutation_test(
    y_true=y_test,
    y_pred_svm=svm_preds,
    y_pred_rf=rf_preds,
    n_permutations=10000
)

# --- 7. Print + plot ---
print(f"\nSVM Accuracy: {svm_acc:.2f}")
print(f"RF Accuracy: {rf_acc:.2f}")
print(f"Observed Accuracy Difference (RF - SVM): {observed_diff:.3f}")
print(f"P-value: {p_val:.4f}")

plt.hist(diff_distribution, bins=30, color='slateblue', edgecolor='black')
plt.axvline(observed_diff, color='red', linestyle='dashed', linewidth=2)
plt.title("Permutation Test: Accuracy Difference (RF - SVM (50,000 - 500,000))")
plt.xlabel("Accuracy Difference")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Optional: Display confusion matrix
cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{ne//1000}k" for ne in ne_values])
disp.plot()
plt.title("RF Confusion Matrix")
plt.show()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# For SVM
cm_svm = confusion_matrix(y_test, svm_preds)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=["50k", "225k", "500k"])
disp_svm.plot(cmap="plasma")  # Optional: pick your color map
plt.title("SVM Confusion Matrix")
plt.show()

