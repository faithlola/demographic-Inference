# demographic-Inference
# MSc Bioinformatics Thesis: Genomic Surveillance of Malaria Vectors in Africa

This repository contains code, scripts, and results from my MSc Bioinformatics thesis on **evolutionary dynamics and genomic surveillance of mosquito populations**, with a focus on **site frequency spectrum (SFS) analysis, msprime simulations, and machine learning approaches** for inferring effective population size (Ne).

---

## 📂 Repository Structure

- **`sfs.py`**  
  Core functions for computing **site frequency spectra (SFS)** from simulated or VCF-based data.  
  - Generates folded/unfolded spectra.  
  - Normalises vectors for input into machine learning models.  

- **`msprime.py`**  
  Scripts for **simulating genomic data** under neutral demographic models using `msprime`.  
  - Simulates allele frequency data under broad and narrow Ne schemes.  
  - Produces SFS vectors and plots for downstream ML analysis.  

- **`svm.py`**  
  Implements a **Support Vector Machine (SVM) classifier** for Ne inference.  
  - Input: Folded, normalised SFS vectors.  
  - Output: Predicted Ne class (broad or narrow).  
  - Uses RBF kernel, trained with 80/20 train-test split.  

- **`random-forest-regressor.py`**  
  Implements a **Random Forest regressor** for continuous Ne estimation.  
  - Input: Folded, normalised SFS vectors.  
  - Output: Continuous Ne predictions.  
  - Includes a mapping step to nearest class bin for direct accuracy comparison with classifiers.  

- **`Permutation-Scenarios.py`**  
  Performs **paired permutation testing** to compare SVM and Random Forest classifiers.  
  - Runs both broad (100k, 500k, 1M) and narrow (50k, 225k, 500k) Ne schemes.  
  - Produces permutation null distributions, p-values, and confusion matrices.  

- **`README.md`**  
  Documentation and usage guide for the repository.  

---

## 🧬 Data Preparation

Population-specific mosquito sample IDs were extracted from the *Anopheles gambiae 1000 Genomes Project (Ag1000G, Phase 2 chromosome 3L variant set)* metadata file.  
- Sample identifiers were extracted and transferred into `Ag1000G_phase2_samples_wild.txt` using the Unix `awk` command.  
- Populations were filtered based on the **country column** and saved into separate text files.  
- VCFs corresponding to each population were downloaded for downstream SFS construction.  
- This was performed on an **HPC environment** to efficiently handle large-scale genomic data.  

Samples were grouped into **West, Central, and East Africa** to allow region-specific comparisons.  
In total:  
- **1,100+ mosquito genomes**  
- **15 distinct collection sites** across **13 countries**  
These sites span areas of high malaria transmission and known insecticide resistance, forming the basis for **SFS analysis and demographic inference**.  

---

## 📊 Workflow

### 1. SFS Construction
- Computed with `scikit-allel`, `numpy`, and `matplotlib`.  
- Genotype data were loaded via `allel.read_vcf()`, then converted to `GenotypeArray`.  
- Allele counts were extracted (`g.count_alleles()`) and filtered to **biallelic 0/1 SNPs** using `is_biallelic_01()`.  
- Histograms of alternate allele counts were plotted to generate **unfolded SFS**.  

### 2. Folded SFS
- Each population was **downsampled to 9 diploids (18 chromosomes)** to standardise sample sizes.  
- Downsampling used Python’s `random.sample()` with a fixed seed (`42`) for reproducibility.  
- Folded SFS constructed by taking the **minimum of allele count and its complement**.  
- Normalised to probability vectors to eliminate SNP count bias.  
- Folded spectra formed the input for machine learning models.  

### 3. Simulation with `msprime`
- Neutral demographic models simulated using `msprime` (via `tskit`).  
- Parameters:  
  - **Population sizes (Ne):** broad (100k, 500k, 1M) and narrow (50k, 225k, 500k)  
  - **Replicates per class:** 5,000  
  - **Chromosomes:** 18 (9 diploids)  
  - **Sequence length:** 10,000 bp  
  - **Mutation rate:** 1.25e-8  
  - **Recombination rate:** 1e-8  
- Folded, normalised SFS vectors were extracted for ML training.  

### 4. SVM Classification
- **SVM (scikit-learn)** trained on folded, normalised spectra.  
- Both broad and narrow schemes tested.  
- Data split 80/20 (train/test).  
- Kernel: **RBF**.  
- Evaluated via accuracy and confusion matrices.  

### 5. Random Forest Regression
- **RandomForestRegressor (scikit-learn)** used to predict continuous Ne values.  
- Allowed inference of intermediate population sizes.  
- Performance assessed with **R², MSE, RMSE** and classification-style mapping.  

### 6. Permutation Testing
- Paired permutation test compared SVM and RF classifiers.  
- Predictions randomly swapped 10,000 times to generate null distributions.  
- Results:  
  - **Broad scheme:** RF 76% vs SVM 74% (p = 0.0005, significant).  
  - **Narrow scheme:** RF 75% vs SVM 74% (p = 0.0934, not significant).  

---

## 🛠 Software Environment

- **Python 3.12.6**  
- **Jupyter Notebook 7.2.2** (miniforge 24.7.1)  
- **msprime 1.3.4**  
- **scikit-allel 1.3.13**  
- **scikit-learn** (SVM, Random Forest)  
- **numpy, matplotlib, tqdm**  

---

## 🚀 Usage

1. **Simulate Data**
   ```bash
   python msprime.py
