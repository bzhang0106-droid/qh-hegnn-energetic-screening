from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
OUT_DIR = ROOT / "manuscript_npJ"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def nrows(rel: str) -> int | str:
    p = ROOT / rel
    if not p.exists():
        return "MISSING"
    if p.suffix.lower() not in {".csv", ".tsv", ".txt", ".md"}:
        return "NA"
    try:
        with p.open(encoding="utf-8", errors="ignore") as fh:
            return max(sum(1 for _ in fh) - 1, 0)
    except Exception:
        return "NA"


def size(rel: str) -> int:
    p = ROOT / rel
    return p.stat().st_size if p.exists() else 0


def exists(rel: str) -> str:
    p = ROOT / rel
    return "OK" if p.exists() and p.stat().st_size > 0 else "MISSING"


def rec(category: str, item: str, rel: str, role: str, placement: str, note: str) -> dict:
    return {
        "Category": category,
        "Item": item,
        "Path": rel,
        "Status": exists(rel),
        "Rows": nrows(rel),
        "Bytes": size(rel),
        "Role": role,
        "Recommended_Manuscript_Placement": placement,
        "Caution_or_Next_Action": note,
    }


records = [
    rec("Core database", "Official frozen AL08 database", "data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv", "Final complete training database", "Data availability / Supplementary Data", "Release sanitized table or DOI repository version; no AL09 planned."),
    rec("Core database", "10D target matrix", "data/baselines/target_matrix_10d.csv", "Complete-case 10-target matrix", "Supplementary Data", "Use as exact modeling target matrix."),
    rec("Core database", "Canonical xTB aligned features", "data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv", "xTB feature table aligned to frozen database", "Supplementary Data / code reproducibility", "Include feature definitions and parsing code."),
    rec("Core database", "AL08 xTB align missing", "data/electronic_features/xtb_align_missing_AL08_20260605.csv", "Confirms xTB missing rows after AL08 alignment", "SI quality-control note", "Rows should be zero."),
    rec("Model", "Final seed42 production metrics", "results/final_model_release/production_v4_10d/final_specialist_10d_bde_metrics.csv", "10D model performance metrics", "Main/SI model performance", "Use final hybrid rows as headline; keep teacher rows for audit."),
    rec("Model", "Final model card", "results/final_model_release/production_v4_10d/PRODUCTION_MODEL_CARD_10D_BDE.txt", "Production-model metadata", "Methods / SI", "Cite AL08, rows=5432, seed42."),
    rec("Model", "Final validation predictions", "manuscript_npJ/SI/model_diagnostics/Supplementary_NPJ_Validation_Predictions_final_specialist_10d_bde_xtbfull_AL08_seed42_20260605.csv", "Per-sample validation true/pred/error audit", "Supplementary Data", "Needed for reproducibility of parity/error analyses."),
    rec("Model", "10D parity figure PNG", "results/model_parity_plots/Figure_Model_Parity_10D_final_specialist_10d_bde_xtbfull_AL08_seed42_20260605.png", "Manuscript-ready 2x5 validation parity figure", "Main Figure or SI Figure", "Use PDF for final submission if possible."),
    rec("P1", "Redundancy/leakage audit table", "manuscript_npJ/SI/model_diagnostics/Table_NPJ_Redundancy_Leakage_Audit_10D_AL08_final.csv", "Nearest-train similarity vs validation error summary", "SI diagnostics", "Supports applicability-domain discussion."),
    rec("P1", "Error-vs-similarity figure", "manuscript_npJ/SI/model_diagnostics/Figure_NPJ_Error_vs_Similarity_10D_AL08_final.png", "10D validation error versus scaffold similarity", "SI Figure", "Use to pre-empt leakage/redundancy reviewer concerns."),
    rec("P2", "Multi-seed stability summary", "manuscript_npJ/SI/model_diagnostics/Table_NPJ_MultiSeed_Stability_10D_AL08_final.csv", "Seed7/42/123 mean and standard deviation", "SI table, Methods robustness", "Use as model stability evidence."),
    rec("P2", "Multi-seed long table", "manuscript_npJ/SI/model_diagnostics/Table_NPJ_MultiSeed_Stability_10D_AL08_final_long.csv", "Per-seed target metrics", "Supplementary Data", "Keep for transparent review."),
    rec("P3", "Teacher-vs-residual table", "manuscript_npJ/SI/model_diagnostics/Table_NPJ_Teacher_vs_Residual_10D_AL08_final.csv", "Final model gain over teacher baseline", "SI model ablation", "Use to justify residual EGNN layer."),
    rec("P3", "Teacher-vs-residual figure", "manuscript_npJ/SI/model_diagnostics/Figure_NPJ_Teacher_Residual_Gain_10D_AL08_final.png", "Delta R2 by target", "SI Figure", "Avoid overstating small/negative gains."),
    rec("P4", "Top20 ranking stability table", "results/ranking_stability/Table_AL08_Final_Top20_Ranking_Stability_10D.csv", "Top20 rank interval under density-bootstrap perturbation", "SI ranking robustness", "Use with density uncertainty; not a synthesis proof."),
    rec("P4", "Rank correlation table", "results/ranking_stability/Table_AL08_Final_Rank_Correlation_10D.csv", "Bootstrap rank-correlation and overlap", "SI ranking robustness", "Summarize top20 overlap in text."),
    rec("P4", "Top20 rank stability figure", "results/ranking_stability/Figure_AL08_Final_Top20_Rank_Stability_10D.png", "Reported rank versus bootstrap rank interval", "SI Figure", "Use to temper exact-rank claims."),
    rec("P5", "Density calibration bootstrap", "results/density_calibration/Table_AL08_Density_Calibration_Bootstrap_10D.csv", "Bootstrap slope/intercept uncertainty for density calibration", "SI uncertainty analysis", "Small n=8 reference set; report as sensitivity analysis."),
    rec("P5", "Top20 detonation uncertainty table", "results/density_calibration/Table_AL08_Top20_Density_Detonation_Uncertainty_10D.csv", "Density-calibration uncertainty propagated to D/P", "SI uncertainty analysis", "Use intervals, not single-rank certainty."),
    rec("P5", "Density uncertainty figure", "results/density_calibration/Figure_AL08_Density_Calibration_Uncertainty_10D.png", "Top20 D intervals from density calibration", "SI Figure", "Figure legend must define bootstrap."),
    rec("P6", "Top20 synthesizability table", "results/synthesizability_10d/Table_AL08_20260605_Top20_Synthesizability_10D.csv", "SA score, chemistry flags, and screening-level synthesis evidence", "SI table; main text only as cautious discussion", "Not sufficient alone to claim demonstrated synthesis."),
    rec("P6", "Top10 retrosynthesis assessment", "results/synthesizability_10d/Table_AL08_20260605_Top10_Retrosynthesis_Assessment_10D.csv", "Rule-based retrosynthetic plausibility notes", "SI table", "Needs manual/literature/route-tool validation for selected lead molecules."),
    rec("P6", "Top20 benchmark similarity table", "results/synthesizability_10d/Table_AL08_20260605_Top20_BenchmarkSimilarity_10D.csv", "Similarity of top molecules to known energetic benchmarks", "SI table", "Use as contextual evidence, not proof of synthesizability."),
    rec("P6", "Top10 retrosynthesis input SMILES", "results/synthesizability_10d/AL08_20260605_Top10_Retrosynthesis_Input.smi", "Input file for external retrosynthesis tools", "Supplementary Data / next-step route planning", "Run ASKCOS/AiZynthFinder or manual route curation before strong synthesis claims."),
    rec("P6", "SA distribution figure", "results/synthesizability_10d/Figure_AL08_20260605_Top20_SA_Distribution_10D.png", "Top20 SA score distribution", "SI Figure", "SA is a screening descriptor only."),
    rec("P6", "Benchmark similarity figure", "results/synthesizability_10d/Figure_AL08_20260605_Top20_BenchmarkSimilarity_10D.png", "Top20 benchmark similarity heatmap", "SI Figure", "Useful for chemical-context discussion."),
    rec("Final report", "AL08 freeze completion report", "manuscript_npJ/final_AL08_freeze_completion_report_20260605.md", "End-to-end final status report", "Internal/manuscript planning", "Use as claim-evidence map before drafting."),
]

df = pd.DataFrame(records)
csv_out = OUT_DIR / "AL08_final_data_inventory_20260605.csv"
md_out = OUT_DIR / "AL08_final_data_inventory_20260605.md"
p6_out = OUT_DIR / "P6_synthesizability_AL08_manifest_20260605.md"
df.to_csv(csv_out, index=False)

db = pd.read_csv(ROOT / "data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv")
prefix = {}
for mol in db["Molecule"].astype(str):
    k = mol.split("_Target_")[0] if "_Target_" in mol else "BASE"
    prefix[k] = prefix.get(k, 0) + 1

ms = pd.read_csv(ROOT / "manuscript_npJ/SI/model_diagnostics/Table_NPJ_MultiSeed_Stability_10D_AL08_final.csv")
metrics = pd.read_csv(ROOT / "results/final_model_release/production_v4_10d/final_specialist_10d_bde_metrics.csv")
final = metrics[metrics["Model_Group"].eq("Final-Specialist-Hybrid-v2")][["Target", "R2", "MAE", "RMSE"]]

ready_notes = [
    "Data package is sufficient for drafting a serious npj Computational Materials manuscript and for internal pre-submission review.",
    "Before journal submission, deposit/share the sanitized data, code, model artifacts, and minimal reproduction instructions through a persistent repository or provide a reviewer-accessible private link.",
    "Treat density-calibrated detonation ranking as uncertainty-limited because the calibration set has eight benchmark molecules.",
    "Treat P6 synthesizability as screening-level evidence. Strong main-text claims about synthesis should be backed by manual route curation, ASKCOS/AiZynthFinder-style route proposals, or literature/precursor evidence for selected final leads.",
]

with md_out.open("w", encoding="utf-8") as fh:
    fh.write("# AL08 Final Data Inventory\n\n")
    fh.write("## Frozen Core\n\n")
    fh.write(f"- Official database rows: {len(db)}\n")
    fh.write(f"- Prefix composition: {prefix}\n")
    fh.write(f"- Target matrix rows: {nrows('data/baselines/target_matrix_10d.csv')}\n")
    fh.write(f"- Canonical xTB aligned rows: {nrows('data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv')}\n")
    fh.write(f"- AL08 xTB align missing rows: {nrows('data/electronic_features/xtb_align_missing_AL08_20260605.csv')}\n\n")
    fh.write("## Final Seed42 Metrics\n\n")
    fh.write(final.to_markdown(index=False) + "\n\n")
    fh.write("## Multi-Seed Stability\n\n")
    fh.write(ms[["Target", "R2_mean", "R2_std", "R2_min", "R2_max", "Seeds"]].to_markdown(index=False) + "\n\n")
    fh.write("## Inventory\n\n")
    fh.write(df[["Category", "Item", "Path", "Status", "Rows", "Recommended_Manuscript_Placement", "Caution_or_Next_Action"]].to_markdown(index=False) + "\n\n")
    fh.write("## Readiness Assessment\n\n")
    for note in ready_notes:
        fh.write(f"- {note}\n")

p6 = df[df["Category"].eq("P6")].copy()
with p6_out.open("w", encoding="utf-8") as fh:
    fh.write("# P6 Synthesizability Evidence Manifest (AL08 Final)\n\n")
    fh.write("These files are the current final AL08 synthesizability evidence package. They support screening-level discussion, not a demonstrated synthesis route.\n\n")
    fh.write(p6[["Item", "Path", "Status", "Rows", "Role", "Caution_or_Next_Action"]].to_markdown(index=False) + "\n\n")
    fh.write("Recommended next step for a stronger npj response: run the Top10 SMILES through a retrosynthesis tool and manually curate precursor availability / plausible disconnection routes for 3-5 lead molecules.\n")

print(f"[SAVE] {csv_out}")
print(f"[SAVE] {md_out}")
print(f"[SAVE] {p6_out}")
