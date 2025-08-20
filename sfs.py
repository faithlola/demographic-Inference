#IMPORT PACKAGES 

import matplotlib.pyplot as plt
import numpy as np
import allel
import random

callset = allel.read_vcf('/data/scratch/bt24018/vcf/vcf_results_03june25/central_africa/eq_guinea_3L.vcf.gz') #change path to where VCF file for each region is

gt = callset['calldata/GT']

g = allel.GenotypeArray(gt)
ac = g.count_alleles()
print(ac)

ac.max_allele()

ac.is_singleton(allele=1)

# Filter for biallelic 0/1 variants. 
is_biallelic_01 = ac.is_biallelic_01(min_mac=None)  # Returns boolean array[8].

# Use the mask to filter the allele counts keeping only 0 and 1.
ac_biallelic_01 = ac[is_biallelic_01]

# For SFS, you typically want the count of the alternate (allele 1).
#Extracts alternative allele count for sfs.
act = ac_biallelic_01[:, 1]
#If i want to exclude rare variants 'is_biallelic_01 = ac.is_biallelic_01(min_mac=2)'

# Now 'alt_counts' can be used to plot the site frequency spectrum

act_nonzero = act[(act > 0) & (act < 2 * g.n_samples)]  
#act = 0 means no alternate alleles. 
#act == 2*g.n_samples means all individuals are homozygous alternate.
#This keeps only polymorphic sites.

# Count frequency of each allele count and calculate site frequency spectrum.
sfs = np.bincount(act_nonzero)
x = np.arange(1, len(sfs))  # Exclude 0 (monomorphic sites).

#UNFOLDED SITE FREQUENCY SPECTRA

plt.bar(x, sfs[1:], width=1.0, edgecolor='black')
plt.xlabel("Alternate Allele Count")
plt.ylabel("Number of sites")
plt.title("Site Frequency Spectrum (SFS) for EQ")
plt.savefig("sfs_eq.png")  # Save the figure as a PNG file
plt.show()

#FOLDED SITE FREQUENCY SPECTRA

# Load VCF and Genotype Data
#callset = allel.read_vcf('/data/scratch/bt24018/vcf/vcf_results_03june25/central_africa/eq_guinea_3L.vcf.gz')
gt = allel.GenotypeArray(callset['calldata/GT'])  # shape: (n_sites, n_samples, ploidy)

# Downsample to 9 diploid mosquitoes (18 chromosomes)
random.seed(42)  # for reproducibility
n_diploids = 9
if gt.n_samples < n_diploids:
    raise ValueError(f"Not enough samples to downsample to {n_diploids} diploids.")
subset_indices = sorted(random.sample(range(gt.n_samples), n_diploids))
gt_sub = gt[:, subset_indices]

# Count alleles at each site
ac = gt_sub.count_alleles()

# Filter for biallelic 0/1 sites
is_biallelic_01 = ac.is_biallelic_01(min_mac=1)
ac_biallelic = ac[is_biallelic_01]

# Get alternate allele count
act = ac_biallelic[:, 1]
n_chroms = 2 * n_diploids  # 20 chromosomes

# Keep only polymorphic sites
act_poly = act[(act > 0) & (act < n_chroms)]

# Fold the SFS: minor allele count
folded = np.minimum(act_poly, n_chroms - act_poly)
sfs = np.bincount(folded)

# Plot
x = np.arange(1, len(sfs))  # skip monomorphic (0)
plt.figure(figsize=(8, 5))
plt.bar(x, sfs[1:], width=1.0, edgecolor='black')
plt.xlabel("Minor Allele Count (Folded)")
plt.ylabel("Number of Sites")
plt.title("Folded Site Frequency Spectrum (EQ Guinea, n=9 diploids)")
plt.tight_layout()
plt.savefig("folded_sfs_eq.png")
plt.show()
