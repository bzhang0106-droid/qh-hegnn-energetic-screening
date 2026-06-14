from __future__ import annotations

from pathlib import Path

import json
import pandas as pd

ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
REPORT = ROOT / "manuscript_npJ/final_AL08_freeze_completion_report_20260605.md"


def nrows(path: Path) -> int | str:
    if not path.exists():
        return "MISSING"
    if path.suffix == ".json":
        return "json"
    with path.open(encoding="utf-8", errors="ignore") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def status(path: str) -> str:
    p = ROOT / path
    return f"`{path}` ({'OK' if p.exists() and p.stat().st_size > 0 else 'MISSING'}, rows={nrows(p)})"


def main() -> None:
    db = pd.read_csv(ROOT / "data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv")
    metrics = pd.read_csv(ROOT / "results/final_model_release/production_v4_10d/final_specialist_10d_bde_metrics.csv")
    final_metrics = metrics[metrics["Model_Group"].eq("Final-Specialist-Hybrid-v2")][["Target", "MAE", "RMSE", "R2"]].copy()
    multiseed = pd.read_csv(ROOT / "manuscript_npJ/SI/model_diagnostics/Table_NPJ_MultiSeed_Stability_10D_AL08_final.csv")
    wf = json.loads((ROOT / "data/workflow_state.json").read_text())

    prefix_counts = {}
    for mol in db["Molecule"].astype(str):
        pref = mol.split("_Target_")[0] if "_Target_" in mol else "BASE"
        prefix_counts[pref] = prefix_counts.get(pref, 0) + 1

    excluded = pd.DataFrame(
        [
            ["AL08_Target_0001", "ORCA geometry optimization failure"],
            ["AL08_Target_0152", "ORCA geometry optimization failure"],
            ["AL08_Target_0156", "ORCA frequency failure"],
            ["AL08_Target_0203", "ORCA geometry optimization failure"],
            ["AL08_Target_0275", "ORCA frequency failure"],
            ["AL08_Target_0378", "Multiwfn timeout during surface descriptor extraction"],
            ["AL08_Target_0383", "ORCA frequency failure"],
        ],
        columns=["Molecule", "Reason"],
    )
    failed = pd.read_csv(ROOT / "data/final_verification_results_AL08_20260605.csv")
    failed = failed[failed["BDE_Parse_Status"].astype(str).eq("failed")][["Molecule", "SMILES", "BDE_Parse_Status"]]
    excluded_out = ROOT / "manuscript_npJ/SI/model_diagnostics/Table_AL08_Excluded_Or_Failed_Verification_Molecules.csv"
    pd.concat([excluded, failed.assign(Reason="BDE parse failure")], ignore_index=True).to_csv(excluded_out, index=False)

    p_items = [
        ("P1 redundancy/leakage", [
            "manuscript_npJ/SI/model_diagnostics/Figure_NPJ_Error_vs_Similarity_10D_AL08_final.png",
            "manuscript_npJ/SI/model_diagnostics/Table_NPJ_Redundancy_Leakage_Audit_10D_AL08_final.csv",
        ]),
        ("P2 multi-seed stability", [
            "manuscript_npJ/SI/model_diagnostics/Table_NPJ_MultiSeed_Stability_10D_AL08_final.csv",
        ]),
        ("P3 teacher vs residual", [
            "manuscript_npJ/SI/model_diagnostics/Table_NPJ_Teacher_vs_Residual_10D_AL08_final.csv",
            "manuscript_npJ/SI/model_diagnostics/Figure_NPJ_Teacher_Residual_Gain_10D_AL08_final.png",
        ]),
        ("P4 Top20 ranking stability", [
            "results/ranking_stability/Table_AL08_Final_Top20_Ranking_Stability_10D.csv",
            "results/ranking_stability/Figure_AL08_Final_Top20_Rank_Stability_10D.png",
        ]),
        ("P5 density calibration uncertainty", [
            "results/density_calibration/Table_AL08_Density_Calibration_Bootstrap_10D.csv",
            "results/density_calibration/Table_AL08_Top20_Density_Detonation_Uncertainty_10D.csv",
        ]),
        ("P6 synthesizability", [
            "results/synthesizability_10d/Table_AL08_20260605_Top20_Synthesizability_10D.csv",
            "results/synthesizability_10d/Table_AL08_20260605_Top10_Retrosynthesis_Assessment_10D.csv",
        ]),
    ]

    lines = []
    lines.append("# Workflow2.0 AL08 Final-Freeze Completion Report\n")
    lines.append("## Final Database State\n")
    lines.append(f"- Official database rows: {len(db)}")
    lines.append(f"- Prefix composition: {prefix_counts}")
    lines.append(f"- Target matrix rows: {nrows(ROOT / 'data/baselines/target_matrix_10d.csv')}")
    lines.append(f"- Canonical xTB aligned rows: {nrows(ROOT / 'data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv')}")
    lines.append(f"- AL08 xTB align missing rows: {nrows(ROOT / 'data/electronic_features/xtb_align_missing_AL08_20260605.csv')}")
    lines.append(f"- Workflow state: last_completed_iter={wf.get('last_completed_iter')}, next_iter={wf.get('next_iter')}, updated_at={wf.get('updated_at')}")
    lines.append("\n## Final Seed42 Metrics\n")
    lines.append(final_metrics.to_markdown(index=False))
    lines.append("\n## Multi-Seed Stability\n")
    lines.append(multiseed[["Target", "R2_mean", "R2_std", "R2_min", "R2_max", "Seeds"]].to_markdown(index=False))
    lines.append("\n## P1-P6 Completion Matrix\n")
    for name, paths in p_items:
        lines.append(f"- {name}: " + "; ".join(status(p) for p in paths))
    lines.append("\n## AL08 Excluded Or Failed Molecules\n")
    lines.append(f"- Exclusion/failure table: `{excluded_out.relative_to(ROOT)}`")
    lines.append("\n## Top-Candidate Evidence Files\n")
    for p in [
        "results/ranking_stability/Table_AL08_Final_Top20_Ranking_Stability_10D.csv",
        "results/density_calibration/Table_AL08_Top20_Density_Detonation_Uncertainty_10D.csv",
        "results/synthesizability_10d/Table_AL08_20260605_Top20_Synthesizability_10D.csv",
        "results/synthesizability_10d/AL08_20260605_Top10_Retrosynthesis_Input.smi",
    ]:
        lines.append(f"- {status(p)}")
    lines.append("\n## Manuscript Claim Readiness\n")
    lines.append("- Strong claims: final 10D model training on the frozen AL08 database; reproducible seed42 metrics; complete target matrix and xTB alignment.")
    lines.append("- Uncertainty-limited claims: density-calibrated detonation ranking, because calibration relies on a small benchmark set and should be presented with bootstrap intervals.")
    lines.append("- SI-supported claims: redundancy/leakage diagnostics, multi-seed stability, teacher-vs-residual audit, and rule-based synthesizability evidence.")
    lines.append("- Cautious claims only: synthetic accessibility. SA score and rule-based disconnection flags are screening evidence, not a demonstrated synthesis route.")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SAVE] {REPORT}")


if __name__ == "__main__":
    main()
