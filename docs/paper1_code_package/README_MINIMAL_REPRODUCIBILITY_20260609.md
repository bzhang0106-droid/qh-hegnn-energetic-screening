# Minimal reviewer reproducibility check

This package includes a lightweight check that avoids licensed quantum-chemistry executables and expensive retraining.

```bash
cd NPJ_submission
python code/minimal_reproduce_reviewer_check.py
```

The script verifies the released curated-database row count, checks the QTAIM-aware Top20 and parity-metric source tables, validates the updated Fig. 1/Fig. 4 source-data files, and writes compact previews under `reviewer_minimal_check/`.

The full workflow scripts are included for transparency, but ORCA, xTB, Multiwfn, Critic2, SLURM and graph-learning dependencies are external installations and may require local path edits.
