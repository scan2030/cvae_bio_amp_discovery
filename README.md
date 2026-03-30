# CVAE-BIO For AMP Discovery

This repository is a code companion for the study:

`Biochemical-knowledge-driven machine learning pipeline for generating potent antimicrobial peptides`

The paper describes `CVAE-BIO`, a multi-module framework for antimicrobial peptide discovery against drug-resistant `Escherichia coli`, combining:

- Module I: biochemical-knowledge-driven peptide generation with a conditional variational autoencoder (CVAE)
- Module II: antimicrobial activity prediction with a Random Forest classifier trained on 30 biochemical descriptors
- Module III: wet-lab validation of prioritized candidates

The current repository focuses on the computational code used for sequence generation and downstream biochemical-feature-based analysis, while keeping the original workflow close to the notebook implementation used in the study.

## Citation

If you use this repository, please cite:

- Yang D, Li Y, Li C, et al. `Biochemical-knowledge-driven machine learning pipeline for generating potent antimicrobial peptides`. Briefings in Bioinformatics. 2026;27:bbag115. [https://doi.org/10.1093/bib/bbag115](https://doi.org/10.1093/bib/bbag115)

## Repository structure

- `notebooks/CVAE_BIO_module1_generator.ipynb`: main notebook for the CVAE-based peptide generation workflow.
- `notebooks/APD_peptide_generation_CVAE.ipynb`: legacy notebook filename retained for compatibility.
- `src/features.py`: peptide property and descriptor helpers.
- `src/preprocessing.py`: sequence encoding helpers.
- `src/models.py`: autoencoder, VAE, and conditional VAE definitions.
- `src/training.py`: training loops, latent extraction, and reconstruction helpers.
- `data/README.md`: notes on the APD-derived input data used in the study.
- `CITATION.cff`: citation metadata for GitHub.
- `requirements.txt`: core Python dependencies used by the notebook.

## What was cleaned up

- Kept the notebook as the main deliverable.
- Split the reusable code into `src/` modules while keeping the notebook as the main experiment file.
- Added a minimal project layout for GitHub readability.
- Reframed the repository around the paper terminology (`CVAE-BIO`, modules, biochemical constraints) instead of internal experiment labels.
- Kept the implementation close to the original notebook to avoid changing model behavior unnecessarily.

## Suggested usage

1. Create a Python environment.
2. Install dependencies from `requirements.txt`.
3. Place the APD-derived input table in the project data directory or update the notebook path accordingly.
4. Open and run `notebooks/CVAE_BIO_module1_generator.ipynb`.

## Scope

- The notebook and `src/` modules cover the computational parts of the study.
- Wet-lab validation results reported in the paper are not reproduced by code in this repository.
- The current notebook most closely corresponds to Module I and parts of Module II of the published pipeline.

## Data

- The study uses peptide records derived from APD3 and focused on peptides with documented activity against `E. coli`.
- The repository currently expects the processed CSV input used in the notebook workflow, for example `APD_ac_10_label.csv`.
- See `data/README.md` for a short note on expected data placement and provenance.

## Notes

- The notebook originally included repeated definitions for functions such as `extract_z`, `reconstruct_sequence`, and peptide property calculation. The reusable versions are now collected in `src/`.
- If this repository needs a second pass later, a natural next step would be to split Module II model comparison and downstream filtering into their own scripts.
- The paper is Open Access under CC BY-NC for the article text. Code licensing for the repository should still be confirmed explicitly with the author team before adding a standalone software license file.
