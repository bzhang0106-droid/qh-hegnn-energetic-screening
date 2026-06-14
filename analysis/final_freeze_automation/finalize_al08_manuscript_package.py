from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors, RDConfig

RDLogger.DisableLog("rdApp.*")

try:
    import sys

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
except Exception:  # pragma: no cover - optional RDKit contrib
    sascorer = None


ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
DATE = "20260605"
FINAL_TAG = "AL08_20260605"
MANUSCRIPT = ROOT / "manuscript_npJ"
PACKAGE = MANUSCRIPT / f"final_submission_package_AL08_{DATE}"
P6_DIR = ROOT / "results" / "synthesizability_10d"


BENCHMARKS = {
    "TNT": "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",
    "TATB": "Nc1c(N)c(N)c([N+](=O)[O-])c([N+](=O)[O-])c1[N+](=O)[O-]",
    "FOX-7": "NC(=C([N+](=O)[O-])[N+](=O)[O-])N",
    "RDX": "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])C1",
    "HMX": "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])CN([N+](=O)[O-])C1",
    "PETN": "C(CO[N+](=O)[O-])(CO[N+](=O)[O-])(CO[N+](=O)[O-])CO[N+](=O)[O-]",
    "NTO": "O=c1[nH]n[nH]c(=O)n1",
}

RISK_SMARTS = {
    "C_nitro": "[#6][N+](=O)[O-]",
    "N_nitro_or_nitramine": "[#7][N+](=O)[O-]",
    "Nitrate_ester": "[OX2][N+](=O)[O-]",
    "Azo_or_diazo": "[NX2]=[NX2]",
    "Azide": "[N-]=[N+]=N",
    "Peroxide_like": "[OX2][OX2]",
    "Tetrazole_like": "n1nnnc1",
    "Furazan_like": "o1nncc1",
}

PATTERNS = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in RISK_SMARTS.items()
    if Chem.MolFromSmarts(smarts) is not None
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def mol(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def fp(m):
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m is not None else None


def murcko_smiles(m) -> str:
    if m is None:
        return ""
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaf = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(scaf) if scaf is not None and scaf.GetNumAtoms() else ""
    except Exception:
        return ""


def sa_score(m) -> float:
    if m is None or sascorer is None:
        return float("nan")
    return float(sascorer.calculateScore(m))


def substructure_counts(m) -> dict[str, int]:
    if m is None:
        return {k: 0 for k in PATTERNS}
    return {k: len(m.GetSubstructMatches(patt)) for k, patt in PATTERNS.items()}


def benchmark_table() -> pd.DataFrame:
    rows = []
    for name, smi in BENCHMARKS.items():
        m = mol(smi)
        rows.append(
            {
                "Benchmark": name,
                "SMILES": smi,
                "Canonical_SMILES": Chem.MolToSmiles(m) if m is not None else "",
                "Murcko_Scaffold": murcko_smiles(m),
                "SA_Score_RDKit": sa_score(m),
                "ExactMolWt": Descriptors.ExactMolWt(m) if m is not None else float("nan"),
                "Ring_Count": rdMolDescriptors.CalcNumRings(m) if m is not None else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def nearest_benchmark_info(m, benches: pd.DataFrame) -> tuple[str, float, str, float]:
    f = fp(m)
    if f is None:
        return "NA", float("nan"), "NA", float("nan")
    best = ("NA", -1.0)
    best_scaf = ("NA", -1.0)
    for _, b in benches.iterrows():
        bm = mol(b["Canonical_SMILES"])
        bf = fp(bm)
        if bf is not None:
            sim = float(DataStructs.TanimotoSimilarity(f, bf))
            if sim > best[1]:
                best = (b["Benchmark"], sim)
        cand_scaf = mol(murcko_smiles(m))
        bench_scaf = mol(str(b["Murcko_Scaffold"]))
        cf, sf = fp(cand_scaf), fp(bench_scaf)
        if cf is not None and sf is not None:
            sim = float(DataStructs.TanimotoSimilarity(cf, sf))
            if sim > best_scaf[1]:
                best_scaf = (b["Benchmark"], sim)
    return best[0], best[1], best_scaf[0], best_scaf[1]


def evidence_tier(row: pd.Series, sa_benchmark_q75: float) -> tuple[str, str, str, str]:
    sa = row["SA_Score_RDKit"]
    sim = row["Nearest_Benchmark_Tanimoto"]
    scaf_sim = row["Nearest_Benchmark_Scaffold_Tanimoto"]
    high_risk = bool(row["High_Risk_Motif_Count"] > 0)
    unsupported_novel = bool(sim < 0.25 and scaf_sim < 0.25)
    very_high_sa = bool(sa > max(5.5, sa_benchmark_q75 + 1.0))

    if (sim >= 0.45 or scaf_sim >= 0.55) and sa <= max(5.0, sa_benchmark_q75) and not high_risk:
        tier = "A"
        use = "Main text candidate if property ranking remains strong"
        claim = "Known-motif/benchmark-adjacent and RDKit-SA-compatible screening evidence"
        action = "Still cite as computational recommendation; route validation desirable for lead."
    elif (sim >= 0.25 or scaf_sim >= 0.35) and sa <= 5.5 and not very_high_sa:
        tier = "B"
        use = "SI candidate; cautious main-text mention only as screened lead"
        claim = "Plausible energetic-motif analogue with moderate synthetic-accessibility evidence"
        action = "Run retrosynthesis tool and check precursor/literature availability before strong claim."
    elif unsupported_novel or very_high_sa or high_risk:
        tier = "C"
        use = "SI only; not a manuscript lead without independent route evidence"
        claim = "Novel/high-risk screening hit; synthesizability not established"
        action = "Manual route curation required; deprioritize if route search fails."
    else:
        tier = "B/C"
        use = "SI only unless external route evidence is added"
        claim = "Mixed screening evidence"
        action = "Resolve by ASKCOS/AiZynthFinder/manual precursor search."
    return tier, use, claim, action


def route_hypothesis(row: pd.Series) -> tuple[str, str]:
    if row["N_nitro_or_nitramine"] > 0:
        return (
            "late-stage N-nitration/nitramine formation after assembling the N-rich core",
            "amino, hydrazino, aminal, or ring-N precursor; verify nitration selectivity",
        )
    if row["Nitrate_ester"] > 0:
        return (
            "late-stage nitrate ester formation from hydroxylated precursor",
            "polyhydroxy/hydroxymethyl precursor; verify thermal and handling stability",
        )
    if row["C_nitro"] > 0:
        return (
            "late-stage C-nitration of activated heteroaromatic or carbocyclic precursor",
            "activated arene/heteroarene; verify positional selectivity and overnitration risk",
        )
    if row["Tetrazole_like"] > 0 or row["Furazan_like"] > 0:
        return (
            "construct N/O-rich heterocycle first, then install energetic substituents",
            "furazan/tetrazole/triazole-like precursor; verify ring formation precedent",
        )
    if row["Azo_or_diazo"] > 0:
        return (
            "oxidative coupling or diazotization-like formation of N=N linkage",
            "amino N-rich precursor; verify azo coupling and sensitivity",
        )
    return (
        "core-first assembly followed by oxidation/nitration where compatible",
        "route not obvious from substructure rules; external retrosynthesis required",
    )


def build_strict_p6() -> dict[str, Path]:
    src = P6_DIR / "Table_AL08_20260605_Top20_Synthesizability_10D.csv"
    route_src = P6_DIR / "Table_AL08_20260605_Top10_Retrosynthesis_Assessment_10D.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    top = pd.read_csv(src)
    route = pd.read_csv(route_src) if route_src.exists() else pd.DataFrame()
    benches = benchmark_table()
    q75 = float(benches["SA_Score_RDKit"].quantile(0.75))
    q50 = float(benches["SA_Score_RDKit"].median())

    records = []
    for _, row in top.iterrows():
        m = mol(row["SMILES"])
        scaf = murcko_smiles(m)
        nearest, sim, nearest_scaf, scaf_sim = nearest_benchmark_info(m, benches)
        counts = substructure_counts(m)
        high_risk = int(counts.get("Peroxide_like", 0) + counts.get("Azide", 0))
        route_h, precursor = route_hypothesis(pd.Series({**row.to_dict(), **counts}))
        rec = {
            "Rank_by_Oracle_D": row.get("Rank_by_Oracle_D"),
            "Molecule": row.get("Molecule"),
            "SMILES": row.get("SMILES"),
            "Canonical_SMILES": Chem.MolToSmiles(m) if m is not None else "",
            "Murcko_Scaffold": scaf,
            "Oracle_D(km/s)": row.get("Oracle_D(km/s)"),
            "Oracle_P(GPa)": row.get("Oracle_P(GPa)"),
            "Vertical_BDE(kcal/mol)": row.get("Vertical_BDE(kcal/mol)"),
            "SA_Score_RDKit": row.get("SA_Score_RDKit", sa_score(m)),
            "SA_Benchmark_Median": q50,
            "SA_Benchmark_Q75": q75,
            "ExactMolWt": Descriptors.ExactMolWt(m) if m is not None else float("nan"),
            "cLogP_RDKit": Crippen.MolLogP(m) if m is not None else float("nan"),
            "TPSA_RDKit": rdMolDescriptors.CalcTPSA(m) if m is not None else float("nan"),
            "Ring_Count": rdMolDescriptors.CalcNumRings(m) if m is not None else float("nan"),
            "Nearest_Benchmark": nearest,
            "Nearest_Benchmark_Tanimoto": sim,
            "Nearest_Benchmark_Scaffold": nearest_scaf,
            "Nearest_Benchmark_Scaffold_Tanimoto": scaf_sim,
            "High_Risk_Motif_Count": high_risk,
            "Route_Hypothesis": route_h,
            "Likely_Precursor_Family": precursor,
        }
        rec.update(counts)
        tier, use, claim, action = evidence_tier(pd.Series(rec), q75)
        rec["Evidence_Tier"] = tier
        rec["Recommended_Manuscript_Use"] = use
        rec["Allowed_Claim"] = claim
        rec["Required_Next_Action"] = action
        rec["Reviewer_Risk"] = (
            "high" if tier == "C" else "medium" if tier == "B/C" else "moderate" if tier == "B" else "lower"
        )
        records.append(rec)

    strict = pd.DataFrame(records)
    strict_path = P6_DIR / "Table_AL08_20260605_P6_Strict_Synthesizability_Evidence_10D.csv"
    strict.to_csv(strict_path, index=False)

    route_check = strict.head(10)[
        [
            "Rank_by_Oracle_D",
            "Molecule",
            "SMILES",
            "Evidence_Tier",
            "Nearest_Benchmark",
            "Nearest_Benchmark_Tanimoto",
            "Nearest_Benchmark_Scaffold",
            "Nearest_Benchmark_Scaffold_Tanimoto",
            "SA_Score_RDKit",
            "Route_Hypothesis",
            "Likely_Precursor_Family",
            "Required_Next_Action",
        ]
    ].copy()
    route_check["External_Retrosynthesis_Status"] = "not_run"
    route_check["Manual_Literature_Precursor_Status"] = "not_checked"
    route_check["Submission_Claim_Status"] = "screening_only_until_route_validated"
    route_path = P6_DIR / "Table_AL08_20260605_Top10_Route_Validation_Checklist_10D.csv"
    route_check.to_csv(route_path, index=False)

    bench_path = P6_DIR / "Table_AL08_20260605_Benchmark_SA_Calibration_10D.csv"
    benches.to_csv(bench_path, index=False)

    query_path = P6_DIR / "AL08_20260605_Top10_External_Retrosynthesis_Query.smi"
    with query_path.open("w", encoding="utf-8") as fh:
        for _, r in route_check.iterrows():
            fh.write(f"{r['SMILES']} {r['Molecule']}\n")

    colors = {"A": "#2F6B4F", "B": "#4D78A8", "B/C": "#B5822A", "C": "#9C3D35"}
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for tier, sub in strict.groupby("Evidence_Tier", sort=False):
        ax.scatter(
            sub["SA_Score_RDKit"],
            sub["Nearest_Benchmark_Tanimoto"],
            s=44,
            color=colors.get(tier, "#777777"),
            label=f"Tier {tier}",
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axvline(q75, color="#444444", linestyle="--", linewidth=0.8, label="benchmark SA Q75")
    ax.axhline(0.25, color="#777777", linestyle=":", linewidth=0.8)
    ax.axhline(0.45, color="#777777", linestyle=":", linewidth=0.8)
    ax.set_xlabel("RDKit synthetic accessibility score")
    ax.set_ylabel("Nearest benchmark Tanimoto similarity")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    tier_fig_png = P6_DIR / "Figure_AL08_20260605_P6_SA_vs_BenchmarkSimilarity_10D.png"
    tier_fig_pdf = P6_DIR / "Figure_AL08_20260605_P6_SA_vs_BenchmarkSimilarity_10D.pdf"
    fig.savefig(tier_fig_png, dpi=300, bbox_inches="tight")
    fig.savefig(tier_fig_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ordered = strict.sort_values("Rank_by_Oracle_D")
    ax.bar(
        ordered["Molecule"].astype(str),
        ordered["SA_Score_RDKit"],
        color=[colors.get(t, "#777777") for t in ordered["Evidence_Tier"]],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(q75, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_ylabel("RDKit SA score")
    ax.set_xlabel("Top-20 rank")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels([str(int(x)) for x in ordered["Rank_by_Oracle_D"]], fontsize=7)
    fig.tight_layout()
    tier_bar_png = P6_DIR / "Figure_AL08_20260605_P6_Evidence_Tier_Map_10D.png"
    tier_bar_pdf = P6_DIR / "Figure_AL08_20260605_P6_Evidence_Tier_Map_10D.pdf"
    fig.savefig(tier_bar_png, dpi=300, bbox_inches="tight")
    fig.savefig(tier_bar_pdf, bbox_inches="tight")
    plt.close(fig)

    methods = MANUSCRIPT / "P6_Strict_Synthesizability_Methods_and_Limitations_AL08_20260605.md"
    tier_counts = strict["Evidence_Tier"].value_counts().to_dict()
    methods.write_text(
        "\n".join(
            [
                "# P6 Strict Synthesizability Evidence (AL08 Final)",
                "",
                "This package replaces the earlier screening-only P6 summary for manuscript planning.",
                "It is designed to support cautious, reviewer-facing statements about synthetic plausibility, not to prove experimental synthesizability.",
                "",
                "## Inputs",
                f"- Top-20 AL08 screened candidates: `{rel(src)}`",
                f"- Existing rule-based route table: `{rel(route_src)}`",
                "- Benchmark energetic materials encoded in the script: TNT, TATB, FOX-7, RDX, HMX, PETN, NTO.",
                "",
                "## Evidence Criteria",
                "- RDKit SA score is compared against the benchmark energetic-material distribution.",
                "- Morgan fingerprint Tanimoto similarity and Murcko-scaffold similarity are computed against benchmark molecules.",
                "- SMARTS rules flag nitro/nitramine, nitrate ester, azo/diazo, azide, peroxide-like, tetrazole-like, and furazan-like motifs.",
                "- Evidence tiers are assigned conservatively: A = benchmark-adjacent and SA-compatible; B = plausible analogue requiring route validation; C = high-risk or unsupported novel hit.",
                "",
                "## Current Tier Counts",
                json.dumps(tier_counts, indent=2, ensure_ascii=False),
                "",
                "## Manuscript Boundary",
                "For npj-style claims, these results may support a statement that selected candidates are computationally prioritized with synthetic-plausibility filters.",
                "They should not be used to claim that the molecules are experimentally synthesizable until external retrosynthesis-tool output, precursor availability, or literature route evidence is added for selected leads.",
                "",
                "## Key Outputs",
                f"- `{rel(strict_path)}`",
                f"- `{rel(route_path)}`",
                f"- `{rel(query_path)}`",
                f"- `{rel(tier_fig_png)}`",
                f"- `{rel(tier_bar_png)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "strict": strict_path,
        "route_check": route_path,
        "bench": bench_path,
        "query": query_path,
        "fig_scatter": tier_fig_png,
        "fig_scatter_pdf": tier_fig_pdf,
        "fig_bar": tier_bar_png,
        "fig_bar_pdf": tier_bar_pdf,
        "methods": methods,
    }


@dataclass
class CopyItem:
    src: str
    dst: str
    role: str
    placement: str


def row_count(path: Path) -> int | str:
    if not path.exists():
        return "MISSING"
    if path.suffix.lower() not in {".csv", ".tsv", ".txt", ".md", ".json", ".smi"}:
        return "NA"
    try:
        with path.open(encoding="utf-8", errors="ignore") as fh:
            return max(sum(1 for _ in fh) - 1, 0)
    except Exception:
        return "NA"


def copy_if_small(src_rel: str, dst_rel: str, max_bytes: int = 50_000_000) -> tuple[str, int]:
    src = ROOT / src_rel
    dst = PACKAGE / dst_rel
    if not src.exists():
        return "MISSING", 0
    if src.stat().st_size > max_bytes:
        return "SKIPPED_LARGE", src.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "COPIED", src.stat().st_size


def build_claim_and_readiness(p6_paths: dict[str, Path]) -> tuple[Path, Path, Path]:
    rows = [
        {
            "Claim_ID": "C1",
            "Claim": "A frozen AL08 database contains 5432 complete 10D energetic-molecule records with xTB alignment complete.",
            "Primary_Evidence": "data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv; data/baselines/target_matrix_10d.csv; data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv",
            "Suggested_Placement": "Methods + Data availability + Supplementary Data",
            "Readiness": "ready_with_repository_release",
            "Boundary": "Do not imply all records are experimental; distinguish calculated, calibrated, and generated/active-learning sources.",
        },
        {
            "Claim_ID": "C2",
            "Claim": "The final 10D specialist model reaches high predictive accuracy across density, thermochemistry, electronic, sensitivity and BDE-related targets.",
            "Primary_Evidence": "results/final_model_release/production_v4_10d/final_specialist_10d_bde_metrics.csv; results/model_parity_plots/Figure_Model_Parity_10D_latest.pdf",
            "Suggested_Placement": "Main results + SI targetwise metrics",
            "Readiness": "ready",
            "Boundary": "Report weaker targets honestly; do not hide VS_max/Sigma2/Nu/BDE moderate R2 values.",
        },
        {
            "Claim_ID": "C3",
            "Claim": "Model performance is robust to seed variation and does not rely on a single seed42 split.",
            "Primary_Evidence": "manuscript_npJ/SI/model_diagnostics/Table_NPJ_MultiSeed_Stability_10D_AL08_final.csv",
            "Suggested_Placement": "SI robustness + brief Methods note",
            "Readiness": "ready",
            "Boundary": "Three seeds are a robustness check, not exhaustive uncertainty quantification.",
        },
        {
            "Claim_ID": "C4",
            "Claim": "Redundancy/leakage was audited through scaffold similarity and validation error-vs-similarity.",
            "Primary_Evidence": "manuscript_npJ/SI/model_diagnostics/Table_NPJ_Redundancy_Leakage_Audit_10D_AL08_final.csv; Figure_NPJ_Error_vs_Similarity_10D_AL08_final.pdf",
            "Suggested_Placement": "SI diagnostic, referenced in Methods",
            "Readiness": "ready",
            "Boundary": "This is a diagnostic audit, not a guarantee of universal out-of-domain generalization.",
        },
        {
            "Claim_ID": "C5",
            "Claim": "The residual/specialist model improves or complements the teacher baseline for most targets.",
            "Primary_Evidence": "manuscript_npJ/SI/model_diagnostics/Table_NPJ_Teacher_vs_Residual_10D_AL08_final.csv",
            "Suggested_Placement": "SI ablation",
            "Readiness": "ready",
            "Boundary": "Call target-specific losses or negligible gains explicitly if present.",
        },
        {
            "Claim_ID": "C6",
            "Claim": "Top AL08 candidate ranking remains interpretable under density-calibration uncertainty.",
            "Primary_Evidence": "results/ranking_stability/Table_AL08_Final_Top20_Ranking_Stability_10D.csv; results/density_calibration/Table_AL08_Top20_Density_Detonation_Uncertainty_10D.csv",
            "Suggested_Placement": "Main/SI candidate ranking",
            "Readiness": "ready_with_caution",
            "Boundary": "Density calibration uses 8 benchmarks; present as sensitivity analysis rather than absolute uncertainty.",
        },
        {
            "Claim_ID": "C7",
            "Claim": "Selected top molecules pass a conservative computational synthesizability screen.",
            "Primary_Evidence": rel(p6_paths["strict"]) + "; " + rel(p6_paths["route_check"]),
            "Suggested_Placement": "SI; cautious main-text sentence only",
            "Readiness": "screening_ready_not_route_proven",
            "Boundary": "No claim of demonstrated synthesis until external retrosynthesis/manual route/literature precursor evidence is added.",
        },
        {
            "Claim_ID": "C8",
            "Claim": "The active-learning loop increases the curated design space and yields final candidate recommendations without starting AL09.",
            "Primary_Evidence": "data/workflow_state.json; final_AL08_freeze_completion_report_20260605.md",
            "Suggested_Placement": "Workflow/results narrative",
            "Readiness": "ready",
            "Boundary": "Keep AL08 as final database freeze; do not mix AL06/AL07 candidates in final claims except as iteration history.",
        },
    ]
    df = pd.DataFrame(rows)
    claim_csv = MANUSCRIPT / f"Claim_Map_AL08_Final_{DATE}.csv"
    claim_md = MANUSCRIPT / f"Claim_Map_AL08_Final_{DATE}.md"
    df.to_csv(claim_csv, index=False)
    claim_md.write_text(
        "# AL08 Final Manuscript/SI Claim Map\n\n"
        + df.to_markdown(index=False)
        + "\n\n"
        "Use this as the controlling ledger when drafting the manuscript and SI. Claims outside this table should be treated as unsupported until linked to an output file.\n",
        encoding="utf-8",
    )

    readiness = pd.DataFrame(
        [
            ["P1 redundancy/leakage", "ready", "Table/Figure present", "SI diagnostic"],
            ["P2 multi-seed stability", "ready", "Seed7/42/123 summary present", "SI robustness"],
            ["P3 teacher vs residual", "ready", "Ablation table/figure present", "SI ablation"],
            ["P4 Top20 ranking stability", "ready_with_caution", "Density bootstrap ranking present", "SI ranking robustness"],
            ["P5 density calibration uncertainty", "ready_with_caution", "8-benchmark bootstrap; sensitivity only", "SI uncertainty"],
            ["P6 synthesizability", "screening_ready_not_route_proven", "Strict P6 rebuilt; external route still needed for strong claim", "SI; cautious main"],
            ["Data availability", "needs_repository", "Frozen DB/target/xTB ready; DOI repository not yet created", "Submission requirement"],
            ["Code availability", "needs_repository_cleanup", "Scripts identified; release README needed", "Submission requirement"],
            ["Safety/dual-use framing", "needs_manuscript_language", "Energetic-material claims must be computational and non-operational", "Main/SI wording"],
        ],
        columns=["Item", "Status", "Evidence_or_Issue", "Placement"],
    )
    readiness_csv = MANUSCRIPT / f"AL08_Final_Readiness_Audit_{DATE}.csv"
    readiness.to_csv(readiness_csv, index=False)
    return claim_csv, claim_md, readiness_csv


def build_peer_benchmark_doc() -> Path:
    path = MANUSCRIPT / f"NPJ_Peer_Evidence_Benchmark_AL08_{DATE}.md"
    path.write_text(
        """# npj Computational Materials Peer-Evidence Benchmark

This note translates comparable npj Computational Materials papers into a checklist for the AL08 manuscript package.

## Comparable Evidence Patterns

1. Active-learning molecular generation papers emphasize a closed-loop workflow, high-throughput quantum validation of generated molecules, self-correction of surrogate errors in new chemical space, and synthesizability checks beyond the score used during generation.
2. Energetic-material de novo design papers show dataset construction and diversity, model comparison tables with MAE/RMSE/R2, Pareto/multi-objective screening, high-precision QM validation of top candidates, and synthetic accessibility evidence benchmarked to known energetic molecules or known scaffolds.
3. GNN/materials-ML papers typically show reproducibility of the modeling workflow, hyperparameter/model comparisons, split strategy, data requirements, and failure modes or limitations.
4. Nature/npj submission guidance expects Data Availability and Code Availability sections, reviewer access to custom code central to claims, high-quality figure files, and supplementary data rather than “data not shown” statements.

## Consequence For This Work

- Main text should focus on the physics-informed active-learning loop, AL08 frozen data growth, 10D prediction accuracy, DFT/ORCA/BDE validation, density-calibrated candidate ranking, and conservative lead recommendation.
- SI should carry the heavy ML diagnostics: multi-seed, leakage/error-vs-similarity, teacher/residual ablation, density-calibration bootstrap, ranking stability, and strict synthesizability evidence.
- P6 should be framed as a computational prioritization filter. Stronger synthesis claims require external retrosynthesis-tool output or manual/literature route curation for selected leads.
""",
        encoding="utf-8",
    )
    return path


def classify_path(path: Path) -> tuple[str, str]:
    r = rel(path)
    suffix = path.suffix.lower()
    size = path.stat().st_size
    if any(part in r for part in ["temp_calc", "xtb_calc", "orca_bde_full_library"]):
        return "exclude_from_repository", "raw/scratch quantum-calculation tree; keep on cluster or archive separately"
    if suffix in {".joblib", ".gbw", ".hess", ".engrad", ".densities", ".densitiesinfo"}:
        return "external_artifact_only", "large model or quantum artifact; register in manifest, store separately if needed"
    if "scripts_archive" in r or ".bak" in r or r.endswith("~"):
        return "archive_obsolete", "historical/back-up script; exclude from clean release"
    if "/logs/" in r or r.startswith("results/logs"):
        return "exclude_logs", "runtime logs; cite only job IDs/status summaries"
    if r.startswith("manuscript_npJ/final_submission_package"):
        return "final_package", "generated final package"
    if r.startswith("manuscript_npJ") or r.startswith("results/model_parity_plots") or r.startswith("results/synthesizability_10d") or r.startswith("results/ranking_stability") or r.startswith("results/density_calibration"):
        return "candidate_manuscript_evidence", "screened for manuscript/SI placement"
    if r.startswith("data/curated_molecule_clean_v1") or r.startswith("data/baselines") or r.startswith("data/electronic_features"):
        return "candidate_supplementary_data", "candidate for repository or Supplementary Data"
    if r.startswith("scripts/") and suffix in {".py", ".sh", ".slurm", ".csv", ".md"}:
        return "candidate_code_release", "script/config for code repository after cleanup"
    if size > 50_000_000:
        return "large_review_before_release", "large file; do not copy blindly"
    return "internal_or_supporting", "not selected for final package unless referenced by claim map"


def screen_all_files() -> tuple[Path, Path]:
    rows = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        action, note = classify_path(p)
        rows.append(
            {
                "Path": rel(p),
                "Bytes": p.stat().st_size,
                "Modified": pd.Timestamp.fromtimestamp(p.stat().st_mtime).isoformat(),
                "Suffix": p.suffix.lower(),
                "Action_Class": action,
                "Note": note,
            }
        )
    df = pd.DataFrame(rows).sort_values(["Action_Class", "Path"])
    out = MANUSCRIPT / f"Workflow2_File_Screening_Manifest_AL08_{DATE}.csv"
    df.to_csv(out, index=False)
    summary = (
        df.groupby("Action_Class")
        .agg(Files=("Path", "count"), Total_GB=("Bytes", lambda x: round(x.sum() / 1e9, 3)))
        .reset_index()
        .sort_values("Action_Class")
    )
    md = MANUSCRIPT / f"Workflow2_File_Screening_Summary_AL08_{DATE}.md"
    md.write_text(
        "# Workflow2.0 File Screening Summary (AL08)\n\n"
        + summary.to_markdown(index=False)
        + "\n\n"
        "This is a metadata-only screening. Large quantum/model artifacts were not read; they are registered for exclusion or external storage.\n",
        encoding="utf-8",
    )
    return out, md


def build_repository_manifest(p6_paths: dict[str, Path]) -> Path:
    rows = [
        ["data", "data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv", "include", "final frozen AL08 database"],
        ["data", "data/baselines/target_matrix_10d.csv", "include", "10D target matrix"],
        ["data", "data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv", "include", "canonical xTB features"],
        ["model", "results/final_model_release/production_v4_10d/final_specialist_10d_bde_metrics.csv", "include", "final model metrics"],
        ["model", "results/final_model_release/production_v4_10d/PRODUCTION_MODEL_CARD_10D_BDE.txt", "include", "model card"],
        ["model", "results/final_model_release/production_v4_10d/final_specialist_10d_bde_production.joblib", "external_large", "4.8 GB model; publish only if repository/storage supports it"],
        ["model", "results/final_model_release/production_v4_10d/seed_stability_AL08/*.joblib", "external_large_or_omit", "seed models are reproducibility aids, not required in SI"],
        ["diagnostics", "manuscript_npJ/SI/model_diagnostics/*AL08*", "include_selected", "final AL08 diagnostics only"],
        ["p6", rel(p6_paths["strict"]), "include", "strict P6 evidence"],
        ["p6", rel(p6_paths["route_check"]), "include", "external route validation checklist"],
        ["code", "scripts/*.py", "include_selected", "canonical workflow scripts; exclude .bak/archive"],
        ["exclude", "temp_calc/, xtb_calc/, orca_bde_full_library/", "exclude", "raw scratch/job directories; too large and not manuscript-ready"],
        ["exclude", "results/logs/", "exclude", "runtime logs; summarize in report only"],
    ]
    df = pd.DataFrame(rows, columns=["Category", "Path_or_Pattern", "Release_Action", "Rationale"])
    out = MANUSCRIPT / f"Repository_Cleanup_Manifest_AL08_{DATE}.csv"
    df.to_csv(out, index=False)
    return out


def assemble_final_package(p6_paths: dict[str, Path], claim_csv: Path, claim_md: Path, readiness_csv: Path, peer_doc: Path, repo_manifest: Path, screen_manifest: Path, screen_summary: Path) -> Path:
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True, exist_ok=True)
    items = [
        CopyItem("results/model_parity_plots/Figure_Model_Parity_10D_latest.pdf", "main_figures/Figure_2_Model_Parity_10D_AL08_seed42.pdf", "10D parity", "main"),
        CopyItem("results/model_parity_plots/Figure_Model_Parity_10D_latest.png", "main_figures/Figure_2_Model_Parity_10D_AL08_seed42.png", "10D parity", "main"),
        CopyItem("results/eda_plots/iter08_02_density_hof_map.png", "main_figures/Figure_3_Density_HOF_Chemical_Space_AL08.png", "chemical-space map", "main/SI"),
        CopyItem("results/ranking_stability/Figure_AL08_Final_Top20_Rank_Stability_10D.pdf", "si_figures/Figure_S_Ranking_Stability_AL08.pdf", "ranking uncertainty", "SI"),
        CopyItem("results/density_calibration/Figure_AL08_Density_Calibration_Uncertainty_10D.pdf", "si_figures/Figure_S_Density_Calibration_Uncertainty_AL08.pdf", "density uncertainty", "SI"),
        CopyItem("manuscript_npJ/SI/model_diagnostics/Figure_NPJ_Error_vs_Similarity_10D_AL08_final.pdf", "si_figures/Figure_S_Error_vs_Similarity_AL08.pdf", "leakage audit", "SI"),
        CopyItem("manuscript_npJ/SI/model_diagnostics/Figure_NPJ_Teacher_Residual_Gain_10D_AL08_final.pdf", "si_figures/Figure_S_Teacher_Residual_Gain_AL08.pdf", "ablation", "SI"),
        CopyItem(rel(p6_paths["fig_scatter_pdf"]), "si_figures/Figure_S_P6_SA_vs_BenchmarkSimilarity_AL08.pdf", "P6 strict evidence", "SI"),
        CopyItem(rel(p6_paths["fig_bar_pdf"]), "si_figures/Figure_S_P6_Evidence_Tier_Map_AL08.pdf", "P6 strict evidence", "SI"),
        CopyItem("results/final_model_release/production_v4_10d/final_specialist_10d_bde_metrics.csv", "main_tables/Table_1_Final_Model_Performance_10D_AL08.csv", "model metrics", "main/SI"),
        CopyItem("results/density_calibration/Table_AL08_Top20_Density_Detonation_Uncertainty_10D.csv", "main_tables/Table_2_Top20_Detonation_Uncertainty_AL08.csv", "candidate uncertainty", "main/SI"),
        CopyItem("manuscript_npJ/SI/model_diagnostics/Table_NPJ_MultiSeed_Stability_10D_AL08_final.csv", "si_tables/Table_S_MultiSeed_Stability_AL08.csv", "robustness", "SI"),
        CopyItem("manuscript_npJ/SI/model_diagnostics/Table_NPJ_Redundancy_Leakage_Audit_10D_AL08_final.csv", "si_tables/Table_S_Redundancy_Leakage_Audit_AL08.csv", "leakage", "SI"),
        CopyItem("manuscript_npJ/SI/model_diagnostics/Table_NPJ_Teacher_vs_Residual_10D_AL08_final.csv", "si_tables/Table_S_Teacher_vs_Residual_AL08.csv", "ablation", "SI"),
        CopyItem("results/ranking_stability/Table_AL08_Final_Top20_Ranking_Stability_10D.csv", "si_tables/Table_S_Top20_Ranking_Stability_AL08.csv", "ranking", "SI"),
        CopyItem(rel(p6_paths["strict"]), "si_tables/Table_S_P6_Strict_Synthesizability_Evidence_AL08.csv", "P6 strict", "SI only until route validation"),
        CopyItem(rel(p6_paths["route_check"]), "si_tables/Table_S_P6_Top10_Route_Validation_Checklist_AL08.csv", "P6 route todo", "SI"),
        CopyItem("manuscript_npJ/SI/model_diagnostics/Table_AL08_Excluded_Or_Failed_Verification_Molecules.csv", "si_tables/Table_S_AL08_Excluded_or_Failed_Verification.csv", "QC", "SI"),
        CopyItem("data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv", "supplementary_data/Supplementary_Data_1_Final_AL08_Database.csv", "database", "repository/SI data"),
        CopyItem("data/baselines/target_matrix_10d.csv", "supplementary_data/Supplementary_Data_2_Target_Matrix_10D_AL08.csv", "target matrix", "repository/SI data"),
        CopyItem("data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv", "supplementary_data/Supplementary_Data_3_xTB_Feature_Matrix_10D_AL08.csv", "xTB features", "repository/SI data"),
        CopyItem("manuscript_npJ/SI/model_diagnostics/Supplementary_NPJ_Validation_Predictions_final_specialist_10d_bde_xtbfull_AL08_seed42_20260605.csv", "supplementary_data/Supplementary_Data_4_Validation_Predictions_AL08_seed42.csv", "validation audit", "repository/SI data"),
        CopyItem(rel(p6_paths["query"]), "supplementary_data/Supplementary_Data_5_P6_Top10_Retrosynthesis_Query.smi", "P6 query input", "repository/SI data"),
    ]
    manifest_rows = []
    for item in items:
        status, size = copy_if_small(item.src, item.dst)
        manifest_rows.append({**item.__dict__, "copy_status": status, "source_bytes": size, "source_rows": row_count(ROOT / item.src)})

    for src in [claim_csv, claim_md, readiness_csv, peer_doc, repo_manifest, screen_manifest, screen_summary, p6_paths["methods"], MANUSCRIPT / "final_AL08_freeze_completion_report_20260605.md"]:
        if src.exists():
            dst = PACKAGE / "internal_audit" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            manifest_rows.append(
                {
                    "src": rel(src),
                    "dst": rel(dst),
                    "role": "internal audit",
                    "placement": "internal",
                    "copy_status": "COPIED",
                    "source_bytes": src.stat().st_size,
                    "source_rows": row_count(src),
                }
            )

    code_dir = PACKAGE / "code_release"
    code_dir.mkdir(parents=True, exist_ok=True)
    code_scripts = [
        "03_egnn_painn_train.py",
        "04_ultimate_discovery.py",
        "05_physics_purifier.py",
        "05b_extract_candidates.py",
        "07_kamlet_jacobs_eval.py",
        "08a_run_multiwfn_critic2.py",
        "08b_ultimate_merge.py",
        "09_eda_analysis.py",
        "11_log_evaluation.py",
        "12_density_calibration_database.py",
        "13b_candidate_space_al_pareto.py",
        "14_ood_trigger_ablation.py",
        "18_build_10d_target_matrix.py",
        "20_prepare_xtb_missing_for_current_db.py",
        "21_finalize_al_iteration.py",
        "23_top_synthesizability_evidence.py",
        "generate_al08_p1_error_vs_similarity.py",
        "generate_al08_p3_p5_evidence.py",
        "generate_al08_p4_ranking_stability.py",
        "summarize_multiseed_al08_20260605.py",
        "finalize_al08_manuscript_package.py",
    ]
    copied = []
    for name in code_scripts:
        src = ROOT / "scripts" / name
        if src.exists() and src.stat().st_size < 2_000_000:
            shutil.copy2(src, code_dir / name)
            copied.append(name)
    (code_dir / "README_CODE_RELEASE_AL08.md").write_text(
        "# AL08 Code Release Draft\n\n"
        "This directory contains the canonical scripts selected for a clean reviewer/repository release. "
        "Historical `.bak` files, Slurm logs, ORCA/xTB scratch directories, and GB-scale joblib models are intentionally excluded. "
        "Before public deposition, add environment details, exact command examples, and a small smoke-test dataset.\n\n"
        "Selected scripts:\n\n"
        + "\n".join(f"- `{x}`" for x in copied)
        + "\n",
        encoding="utf-8",
    )

    display_plan = PACKAGE / "DISPLAY_ITEM_PLAN_AL08_20260605.md"
    display_plan.write_text(
        "# AL08 Display Item Plan\n\n"
        "| Item | Proposed content | Current asset | Status |\n"
        "|---|---|---|---|\n"
        "| Fig. 1 | End-to-end workflow: generator, physics purifier, ORCA/BDE/xTB validation, active learning, final freeze | needs redrawn manuscript schematic | TODO |\n"
        "| Fig. 2 | 10D model parity, two rows x five targets | `main_figures/Figure_2_Model_Parity_10D_AL08_seed42.pdf` | ready |\n"
        "| Fig. 3 | Chemical/property-space expansion and density-HOF map | `main_figures/Figure_3_Density_HOF_Chemical_Space_AL08.png` | candidate, may need panel redesign |\n"
        "| Fig. 4 | Candidate ranking with density/uncertainty and Pareto logic | ranking and density SI figures available | compose main multi-panel |\n"
        "| Fig. 5 | Top leads and conservative P6 screening boundary | P6 strict SI figures available | use only after route validation if making synthesis claims |\n"
        "| SI Fig. Sx | Error vs similarity/leakage | ready | SI |\n"
        "| SI Fig. Sx | Teacher/residual gain | ready | SI |\n"
        "| SI Fig. Sx | Density calibration uncertainty | ready | SI |\n"
        "| SI Fig. Sx | P6 strict evidence tier map | ready, SI only | SI |\n\n"
        "P6 is intentionally not promoted to a main table in the current package because all final Top20 candidates are new-scaffold/high-route-uncertainty under the strict audit.\n",
        encoding="utf-8",
    )

    package_manifest = PACKAGE / "PACKAGE_MANIFEST_AL08_20260605.csv"
    pd.DataFrame(manifest_rows).to_csv(package_manifest, index=False)
    readme = PACKAGE / "README_FINAL_PACKAGE_AL08_20260605.md"
    readme.write_text(
        "# Final AL08 Manuscript Package\n\n"
        "This package is the cleaned AL08 manuscript/SI working set. It intentionally separates final AL08 evidence from older v2, AL06, and AL07 artifacts.\n\n"
        "## Directory Meaning\n"
        "- `main_figures/`: candidate main or high-level figures.\n"
        "- `main_tables/`: compact tables suitable for main text or SI promotion.\n"
        "- `si_figures/` and `si_tables/`: reviewer-facing diagnostics and uncertainty analyses.\n"
        "- `supplementary_data/`: compact release-ready CSV/SMI data.\n"
        "- `code_release/`: draft clean code-release subset.\n"
        "- `internal_audit/`: claim map, file-screening manifest, repository cleanup manifest, and completion report.\n\n"
        "## Important Boundary\n"
        "P6 synthesizability remains computational screening evidence and is kept in SI rather than promoted as a main-text data claim. It is now stricter and manuscript-auditable, but strong synthesis claims still require external retrosynthesis/manual route/literature validation for selected leads.\n",
        encoding="utf-8",
    )
    return package_manifest


def main() -> None:
    MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    p6_paths = build_strict_p6()
    claim_csv, claim_md, readiness_csv = build_claim_and_readiness(p6_paths)
    peer_doc = build_peer_benchmark_doc()
    screen_manifest, screen_summary = screen_all_files()
    repo_manifest = build_repository_manifest(p6_paths)
    package_manifest = assemble_final_package(
        p6_paths,
        claim_csv,
        claim_md,
        readiness_csv,
        peer_doc,
        repo_manifest,
        screen_manifest,
        screen_summary,
    )
    print(json.dumps({
        "p6_strict": rel(p6_paths["strict"]),
        "p6_route_check": rel(p6_paths["route_check"]),
        "claim_map": rel(claim_md),
        "readiness": rel(readiness_csv),
        "peer_benchmark": rel(peer_doc),
        "file_screening": rel(screen_manifest),
        "repo_manifest": rel(repo_manifest),
        "package": rel(PACKAGE),
        "package_manifest": rel(package_manifest),
    }, indent=2))


if __name__ == "__main__":
    main()
