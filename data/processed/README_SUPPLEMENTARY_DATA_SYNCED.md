# Supplementary Data README

This folder contains the final supplementary data files for the npj Computational Materials submission package.

## Final data files

- `Supplementary_Data_1_Curated_5432_Molecule_Database.csv`: curated 5432-molecule database with detonation ranking and prediction fields.
- `Supplementary_Data_2_Target_Matrix_10D.csv`: ten-target matrix used for model training and evaluation.
- `Supplementary_Data_3_xTB_Feature_Matrix_10D.csv`: xTB and low-cost physical feature matrix.
- `Supplementary_Data_4_Validation_Predictions_Final_Hybrid_seed42.csv`: Hybrid validation predictions for seed 42.
- `Supplementary_Data_5_Final_Global_Top20_Retrosynthesis_Input.smi`: Top20 candidate SMILES input for retrosynthesis and manual route review.
- `Supplementary_Data_6_Common_Energetic_Molecules_DLPNoMP2_Enthalpy.csv`: DLPNO-MP2 enthalpy records for the common energetic-molecule seed set used to initialize the scaffold/nitration library.

## Figure and table source files

- `Table_2_QTAIM_Aware_Stability_Constrained_Top20_20260607.csv`: source table for the main-text Top20 comparison table.
- `Table_S_Figure3_HELS_10Target_Parity_Metrics_20260608.csv`: source table for the ten-target parity and validation metric summary.
- `Table_S_QTAIM_Counterfactual_Selection_20260607.csv`: source table for QTAIM-aware counterfactual selection diagnostics.
- `Table_S_QTAIM_Top20_Route_AD_Audit_20260608.csv`: source table for Top20 route-risk and applicability-domain audit.

## Data language

Use the curated 5432-molecule database for the full data set. Top20 candidate files correspond to the QTAIM-aware stability-constrained selection discussed in the manuscript.

## Notes

- Superseded or legacy data snapshots were archived outside the clean submission package.
- The final database row count is 5432 and the first-ranked molecule is `AL04_Target_0027` by `Final_Detonation_D(km/s)`.
