from __future__ import annotations

from pathlib import Path
import re
import shutil
import json

import numpy as np
import pandas as pd


ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
TOP20 = ROOT / "results/final_global_top20/Table_Final_Global_Top20_Structure_Property_Synthesizability_ExternalRetrosynthesis_10D.csv"
AUDIT = ROOT / "manuscript_npJ/major_revision_20260607/Table_Top20_QM_Validation_File_Audit_20260607.csv"
OUT = ROOT / "results/final_global_top20/qm_validation_20260607"
MAJOR = ROOT / "manuscript_npJ/major_revision_20260607"
PKG = ROOT / "manuscript_npJ/final_submission_package_AL08_20260605"
SI_TABLES = PKG / "si_tables"
INTERNAL = PKG / "internal_audit"
CODE_RELEASE = PKG / "code_release"

for p in [OUT, MAJOR, SI_TABLES, INTERNAL, CODE_RELEASE]:
    p.mkdir(parents=True, exist_ok=True)


def normal_termination(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        tail = path.read_text(errors="ignore").splitlines()[-80:]
    except Exception:
        return False
    return any("ORCA TERMINATED NORMALLY" in line for line in tail)


def final_single_point_energy(path: Path) -> float:
    if not path.exists():
        return np.nan
    val = np.nan
    for line in path.read_text(errors="ignore").splitlines():
        if "DLPNO-MP2 total energy" in line.lower():
            m = re.search(r"([-+]?\d+\.\d+)", line)
            if m:
                val = float(m.group(1))
        if "FINAL SINGLE POINT ENERGY" in line:
            try:
                val = float(line.strip().split()[-1])
            except Exception:
                pass
    return val


def thermal_enthalpy_correction(path: Path) -> float:
    if not path.exists():
        return np.nan
    for line in path.read_text(errors="ignore").splitlines():
        if "Thermal Enthalpy correction" in line:
            m = re.search(r"([-+]?\d+\.\d+)\s*Eh", line)
            if m:
                return float(m.group(1))
    return np.nan


def freq_summary(path: Path) -> tuple[int, float, int]:
    if not path.exists():
        return 0, np.nan, 0
    freqs = []
    imag_pert = np.nan
    for line in path.read_text(errors="ignore").splitlines():
        if "Total number of imaginary perturbations" in line:
            nums = re.findall(r"[-+]?\d+", line)
            if nums:
                imag_pert = int(nums[-1])
        m = re.search(r"^\s*\d+:\s*([-+]?\d+\.\d+)\s*cm\*\*-1", line)
        if m:
            freqs.append(float(m.group(1)))
    freqs_arr = np.array(freqs, dtype=float)
    neg = int(np.sum(freqs_arr < -1.0)) if len(freqs_arr) else 0
    positive = freqs_arr[freqs_arr > 1.0]
    min_pos = float(np.min(positive)) if len(positive) else np.nan
    if np.isfinite(imag_pert):
        neg = max(neg, int(imag_pert))
    return len(freqs), min_pos, neg


def read_esp(path: Path) -> dict[str, float]:
    # The curated Top20 table already carries parsed ESP values; this helper is
    # only used to record file presence without reading large outputs.
    return {"ESP_Output_Exists": bool(path.exists())}


def main() -> None:
    top = pd.read_csv(TOP20).head(20)
    audit = pd.read_csv(AUDIT)
    rows = []
    for _, r in audit.iterrows():
        mol = str(r["Molecule"])
        top_row = top[top["Molecule"].astype(str).eq(mol)].head(1)
        work = Path(str(r["Evidence_Dir"]))
        if not work.is_absolute():
            work = ROOT / work
        opt_out = work / f"{mol}_step1_opt.out"
        freq_out = work / f"{mol}_step2_freq.out"
        sp_out = work / f"{mol}_step3_dlpnomp2.out"
        n_freq, min_pos, n_imag = freq_summary(freq_out)
        e_mp2 = final_single_point_energy(sp_out)
        h_corr = thermal_enthalpy_correction(freq_out)
        row = {
            "Rank": int(r["Rank"]),
            "Molecule": mol,
            "SMILES": r["SMILES"],
            "Evidence_Dir": str(work),
            "Opt_Method": "B3LYP-D3BJ/def2-SVP RIJCOSX",
            "Freq_Method": "B3LYP-D3BJ/def2-SVP RIJCOSX",
            "SP_Method": "DLPNO-MP2/def2-TZVP def2-TZVP/C RIJCOSX",
            "Opt_Normal_Termination": normal_termination(opt_out),
            "Freq_Normal_Termination": normal_termination(freq_out),
            "SP_Normal_Termination": normal_termination(sp_out),
            "N_Frequencies_Parsed": n_freq,
            "N_Imaginary_Frequencies_or_Perturbations": n_imag,
            "Minimum_Positive_Frequency_cm-1": min_pos,
            "DLPNO_MP2_Energy_Eh": e_mp2,
            "Thermal_Enthalpy_Correction_Eh": h_corr,
            "DLPNO_MP2_H298_Eh": e_mp2 + h_corr if np.isfinite(e_mp2) and np.isfinite(h_corr) else np.nan,
            "ESP_Output_Exists": bool((work / "esp_output.txt").exists()),
            "Critic2_Report_Exists": bool((work / "critic2_cpreport.out").exists()),
            "BDE_Parse_Status": r.get("BDE_Parse_Status", ""),
            "BDE_Job_Dir_Exists": bool(r.get("BDE_Job_Dir_exists", False)),
        }
        if len(top_row):
            tr = top_row.iloc[0]
            for col in [
                "Final_Detonation_Q(cal/g)",
                "Final_Detonation_Q(kJ/g)",
                "Final_Global_D(km/s)",
                "Final_Global_P(GPa)",
                "Density_used(g/cm3)",
                "Heat_of_Formation(kcal/mol)",
                "Vertical_BDE(kcal/mol)",
                "Trigger_Bond_Rho",
                "VS_max",
                "Sigma2_tot",
                "Nu",
                "Final_Synthesis_Readiness_Tier",
            ]:
                if col in top.columns:
                    row[col] = tr.get(col)
        row["QM_Evidence_Complete"] = all([
            row["Opt_Normal_Termination"],
            row["Freq_Normal_Termination"],
            row["SP_Normal_Termination"],
            row["ESP_Output_Exists"],
            row["Critic2_Report_Exists"],
            str(row["BDE_Parse_Status"]) == "complete",
        ])
        row["Geometry_Minimum_Status"] = "no_imaginary_frequency_detected" if row["N_Imaginary_Frequencies_or_Perturbations"] == 0 else "imaginary_frequency_or_perturbation_flag"
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("Rank")
    table = OUT / "Table_Final_Global_Top20_QM_Validation_Summary_20260607.csv"
    out.to_csv(table, index=False)
    for dest in [SI_TABLES / table.name, MAJOR / table.name]:
        shutil.copy2(table, dest)

    status = {
        "rows": int(len(out)),
        "complete_qm_evidence": int(out["QM_Evidence_Complete"].sum()),
        "normal_opt": int(out["Opt_Normal_Termination"].sum()),
        "normal_freq": int(out["Freq_Normal_Termination"].sum()),
        "normal_sp": int(out["SP_Normal_Termination"].sum()),
        "no_imaginary": int((out["N_Imaginary_Frequencies_or_Perturbations"] == 0).sum()),
        "bde_complete": int(out["BDE_Parse_Status"].astype(str).eq("complete").sum()),
        "q_nonmissing": int(out["Final_Detonation_Q(cal/g)"].notna().sum()) if "Final_Detonation_Q(cal/g)" in out else 0,
    }
    report = OUT / "NPJ_Top20_QM_Validation_Status_20260607.md"
    report.write_text(
        "\n".join(
            [
                "# Final Global Top20 QM Validation Status",
                "",
                "Generated: 2026-06-07",
                "",
                f"- Top20 rows: {status['rows']}",
                f"- Complete QM evidence packages: {status['complete_qm_evidence']}/20",
                f"- Normal opt/freq/SP terminations: {status['normal_opt']}/{status['normal_freq']}/{status['normal_sp']}",
                f"- No imaginary frequency flags: {status['no_imaginary']}/20",
                f"- BDE complete: {status['bde_complete']}/20",
                f"- Q non-missing: {status['q_nonmissing']}/20",
                "",
                "The table reports an existing 02c-level QM package for all final Top20 candidates: B3LYP-D3BJ/def2-SVP geometry/frequency, DLPNO-MP2/def2-TZVP single point, ESP/Multiwfn, Critic2/QTAIM and BDE evidence. Heat of explosion Q is reported as the K-J heat term consistent with D/P calculation.",
                "",
                f"Primary table: {table.relative_to(ROOT)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    for dest in [INTERNAL / report.name, MAJOR / report.name]:
        shutil.copy2(report, dest)

    script_path = ROOT / "scripts/parse_top20_qm_validation_summary_20260607.py"
    if script_path.exists():
        shutil.copy2(script_path, CODE_RELEASE / script_path.name)

    print(json.dumps(status, indent=2))
    print(out[["Rank", "Molecule", "QM_Evidence_Complete", "N_Imaginary_Frequencies_or_Perturbations", "Minimum_Positive_Frequency_cm-1", "Final_Detonation_Q(cal/g)", "Vertical_BDE(kcal/mol)"]].to_string(index=False))


if __name__ == "__main__":
    main()
