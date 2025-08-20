import matplotlib.pyplot as plt
import numpy as np
import msprime
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# --- Function to generate SFS vectors for a given Ne ---
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
        sfs = ts_mut.allele_frequency_spectrum(
            mode="site", polarised=False, span_normalise=False
        )
        sfs_folded = sfs[1:10]  # remove monomorphic bin 0
        if sfs_folded.sum() == 0:
            continue
        sfs_vectors.append(sfs_folded / sfs_folded.sum())

    return np.array(sfs_vectors), snp_counts

# Scenario 1: Broad (100k - 1M)

ne_values = [100_000, 500_000, 1_000_000]
sfs_all, y_all, snp_stats = [], [], {}

for i, ne in enumerate(ne_values):
    sfs_vectors, snp_counts = generate_sfs(ne=ne, n_replicates=5000)
    sfs_all.append(sfs_vectors)
    y_all.extend([i] * len(sfs_vectors))
    snp_stats[ne] = np.mean(snp_counts) if snp_counts else 0.0

X = np.vstack(sfs_all)
y = np.array(y_all)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = SVC(kernel='rbf', random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy (broad):", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{ne//1000}k" for ne in ne_values])
disp.plot()
plt.title("SVM — Ne Class Prediction (Broad)")
Path("results/svm/broad").mkdir(parents=True, exist_ok=True)
plt.savefig("results/svm/broad/confusion_matrix_svm.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nMean SNPs per replicate by Ne (broad):")
for ne in ne_values:
    print(f"Ne = {ne:,}: {snp_stats[ne]:.2f} SNPs")

# Scenario 2: Narrow (50k - 500k)
ne_values = [50_000, 225_000, 500_000]
sfs_all, y_all, snp_stats = [], [], {}

for i, ne in enumerate(ne_values):
    sfs_vectors, snp_counts = generate_sfs(ne=ne, n_replicates=5000)
    sfs_all.append(sfs_vectors)
    y_all.extend([i] * len(sfs_vectors))
    snp_stats[ne] = np.mean(snp_counts) if snp_counts else 0.0

X = np.vstack(sfs_all)
y = np.array(y_all)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = SVC(kernel='rbf', random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy (narrow):", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{ne//1000}k" for ne in ne_values])
disp.plot()
plt.title("SVM — Ne Class Prediction (Narrow)")
Path("results/svm/narrow").mkdir(parents=True, exist_ok=True)
plt.savefig("results/svm/narrow/confusion_matrix_svm.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nMean SNPs per replicate by Ne (narrow):")
for ne in ne_values:
    print(f"Ne = {ne:,}: {snp_stats[ne]:.2f} SNPs")
