# CVAE-BIO For AMP Discovery

This repository accompanies the study:

`Biochemical-knowledge-driven machine learning pipeline for generating potent antimicrobial peptides`

`CVAE-BIO` is a multi-module framework for antimicrobial peptide discovery against drug-resistant `Escherichia coli`, combining:

- Module I: biochemical-knowledge-driven peptide generation with a conditional variational autoencoder (CVAE)
- Module II: antimicrobial activity prediction with a Random Forest classifier trained on 30 biochemical descriptors
- Module III: wet-lab validation of prioritized candidates

This repository contains the computational code used for peptide generation and downstream biochemical-feature-based analysis. The current materials correspond most closely to Module I and parts of Module II of the published pipeline.

## Citation

If you use this repository, please cite:

- Yang D, Li Y, Li C, et al. `Biochemical-knowledge-driven machine learning pipeline for generating potent antimicrobial peptides`. Briefings in Bioinformatics. 2026;27:bbag115. [https://doi.org/10.1093/bib/bbag115](https://doi.org/10.1093/bib/bbag115)

## Repository Contents

- `notebooks/CVAE_BIO_module1_generator.ipynb`: main notebook for the CVAE-based peptide generation workflow.
- `src/features.py`: peptide property and descriptor helpers.
- `src/preprocessing.py`: sequence encoding helpers.
- `src/models.py`: autoencoder, VAE, and conditional VAE definitions.
- `src/training.py`: training loops, latent extraction, and reconstruction helpers.
- `data/README.md`: notes on the APD-derived input data used in the study.
- `requirements.txt`: core Python dependencies used by the notebook.
- `CITATION.cff`: citation metadata for GitHub.

## Quick Start

1. Create a Python environment.
2. Install dependencies from `requirements.txt`.
3. Place the processed APD-derived input table in the project data directory, or update the notebook path accordingly.
4. Open and run `notebooks/CVAE_BIO_module1_generator.ipynb`.

## Data

- The study uses peptide records derived from APD3 and focuses on peptides with documented activity against `E. coli`.
- The notebook expects the processed CSV input used in the workflow, for example `APD_ac_10_label.csv`.
- See `data/README.md` for a short note on expected data placement and provenance.

## Scope

- This repository covers the computational parts of the study.
- Wet-lab validation results reported in the paper are not reproduced by code in this repository.
