# msprime.py
import numpy as np
import matplotlib.pyplot as plt
import msprime

# --- Shared parameters ---
n_diploids = 9
seq_len = 10_000
mut_rate = 1.25e-8
rec_rate = 1e-8
random_seed = 42

def simulate_and_plot(ne, scheme_name):
    """Simulate a single replicate SFS and save a plot."""
    ts = msprime.sim_ancestry(
        samples=2 * n_diploids,
        recombination_rate=rec_rate,
        sequence_length=seq_len,
        population_size=ne,
        random_seed=random_seed
    )

    ts_mut = msprime.sim_mutations(ts, rate=mut_rate, random_seed=random_seed)

    # Compute folded, normalised SFS
    sfs = ts_mut.allele_frequency_spectrum(
        mode="site", polarised=False, span_normalise=False
    )
    sfs_folded = sfs[1: n_diploids + 1]  # remove monomorphic bin
    sfs_scaled = sfs_folded / sfs_folded.sum()

    # Plot
    x = np.arange(1, len(sfs_scaled) + 1)
    plt.bar(x, sfs_scaled, edgecolor='black')
    plt.xlabel("Minor Allele Count (Folded; n=9 diploids)")
    plt.ylabel("Proportion of Sites")
    plt.title(f"Normalised Folded SFS (Ne = {ne:,}, {scheme_name} scheme)")
    plt.tight_layout()
    plt.savefig(f"folded_sfs_{scheme_name}_ne{ne//1000}k.png")
    plt.close()

    # Print vector for inspection
    print(f"Simulated SFS vector (Ne={ne:,}, {scheme_name}):", sfs_scaled.tolist())


# --- SCENARIO 1: Broad (100k – 1M) ---
broad_ne_values = [100_000, 500_000, 1_000_000]
for ne in broad_ne_values:
    simulate_and_plot(ne, scheme_name="broad")

# --- SCENARIO 2: Narrow (50k – 500k) ---
narrow_ne_values = [50_000, 225_000, 500_000]
for ne in narrow_ne_values:
    simulate_and_plot(ne, scheme_name="narrow")


