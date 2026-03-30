# Data Notes

This repository uses an APD3-derived peptide table prepared for the published CVAE-BIO workflow.

Expected columns include:

- `APD ID`
- `Sequence`
- `label`
- `Hydrophobicity`
- `Amphiphilicity`
- `Charge`

The paper states that the underlying peptide records were obtained from APD3 and filtered for peptides with documented activity against `E. coli`.

Recommended practice:

- Keep raw downloaded data separate from processed analysis tables.
- Do not commit large intermediate outputs unless they are needed for reproducibility.
- If data redistribution is restricted, commit only scripts and documentation, not the raw exported database tables.
