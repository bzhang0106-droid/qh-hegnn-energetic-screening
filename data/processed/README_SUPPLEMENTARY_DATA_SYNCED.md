# Supplementary Data Package

Synced on 2026-06-26 against the current manuscript and Supplementary Information.

## Scope

This folder contains the processed data needed to support the main-text figures, SI tables, and the repaired true-phys/HGS Fig. 4 robustness analysis. It intentionally excludes raw ORCA/Multiwfn/Critic2 working directories, Slurm logs, temporary split JSON files, failed OOD exploration outputs, and raw `parts/` CSV files. Raw `parts/` files are not final-scale labels and should not be used for training or manuscript claims.

## Core manuscript data

- `Supplementary_Data_1_Curated_5432_Molecule_Database.csv`: standardized 5432-molecule CHNO library used by the manuscript.
- `Supplementary_Data_2_Target_Matrix_10D.csv`: formal ten-target matrix for the 5432-molecule library.
- `Supplementary_Data_3_xTB_Feature_Matrix_10D.csv`: aligned xTB feature matrix for the formal 10D task.
- `Supplementary_Data_4_Validation_Predictions_Final_Hybrid_seed42.csv`: validation predictions supporting the Fig. 3 observed-predicted parity panels and target-level error summary.
- `Supplementary_Data_5_Final_Global_Top20_Retrosynthesis_Input.smi`: Top20 SMILES input for retrosynthesis checks.
- `Supplementary_Data_6_Common_Energetic_Molecules_DLPNoMP2_Enthalpy.csv`: DLPNO-MP2 enthalpy records for the common energetic-molecule seed set.

## Repaired Fig. 4 and true-phys/HGS data

- `TruePhys_FinalScale_Valid10D_Labels_20260623.csv`: final-scale valid10d labels from the active-learning repair step; raw calculation parts are not included.
- `TruePhys_SurfaceESP_QTAIM_Features_20260624.csv`: merged surface-ESP/QTAIM feature table used by the repaired HGS evaluation.
- `TruePhys_Feature_Coverage_20260624.csv`, `TruePhys_Feature_Status_20260624.csv`, and `TruePhys_Feature_Manifest_QA_20260624.csv`: coverage and source QA for the true-phys feature layer.
- `Model_Robustness_Validation_Predictions_Long.csv`, `Model_Robustness_Validation_R2_Summary.csv`, `Model_Robustness_Validation_Metrics_Long.csv`, `Model_Robustness_Validation_Rows.csv`, `Model_Robustness_Validation_Split_Inventory.csv`, and `Model_Robustness_Validation_Run_Completion.json`: three-split, three-seed repaired Fig. 4 evaluation records. The repaired external-validation set covers Density, Heat_of_Formation, HOMO_LUMO_Gap, SA_Score, VS_max, Sigma2_tot, Nu, Trigger_Bond_Rho, and Vertical_BDE; Molecular_Weight is not part of the repaired Fig. 4 external-validation target set.
- `Source_Data_Model_Robustness_Applicability.csv`: plotted source data for main-text Fig. 4.

## Figure and table source data

- `Source_Data_Screening_Workflow.csv`: compact source data for the screening workflow figure.
- `Source_Data_Figure_3_Parity_Metrics.csv`: target-level parity/model metrics supporting the Fig. 3 validation summary.
- `Source_Data_Table_2_QTAIM_Aware_Top20.csv`: source rows for the QTAIM-aware Top20 table.
- `parsed_quantum_records/`: parsed QTAIM/frequency records and manifest for Top20 quantum-chemistry follow-up evidence. Large binary quantum scratch files are not redistributed.

## Manifest

`MANIFEST_SUPPLEMENTARY_DATA_20260626.csv` records the role, manuscript/SI support, source path, size, row/column counts where applicable, and SHA-256 checksum for each retained file.
