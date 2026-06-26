#!/usr/bin/env python3
"""Build true surface-ESP and QTAIM/rhoBCP intermediate feature tables.

This script is intentionally a working-copy post-processing tool. It reads the
frozen 10d training matrix and existing ORCA wavefunction files, then writes a
separate feature table. It does not modify the formal submission package.

Three subcommands are provided:

  prepare-manifest  Map each molecule to an existing molden or gbw source.
  extract           Run/parse Multiwfn for one chunk of manifest rows.
  summarize         Merge chunk outputs and write compact QA tables.

Target-leakage note:
The table keeps direct surface values such as phys_esp_vs_max_kcal and
phys_esp_sigma2_tot_au2 because they are useful for QA and for optional direct
oracle baselines. Downstream ML must mask the direct target-equivalent columns
when evaluating a surrogate for VS_max, Sigma2_tot, Nu, or Trigger_Bond_Rho.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_DATASET = Path(
    os.environ.get(
        "QH_HEGNN_DATASET",
        "/scratch/gma/bzhang/qh_hegnn_standard_validation/data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv",
    )
)
DEFAULT_ACTIVE_ROOT = Path(
    os.environ.get("QH_HEGNN_ACTIVE_ROOT", "/scratch/gma/bzhang/qh_hegnn_active_learning_batch")
)
DEFAULT_OUT_ROOT = Path(
    os.environ.get("QH_HEGNN_PHYSICS_FEATURE_ROOT", "/scratch/gma/bzhang/qh_hegnn_physics_features")
)
DEFAULT_MULTIWFN = Path("/home/gma/bzhang/soft/Multiwfn_2026.2.2_bin_Linux_noGUI/Multiwfn_noGUI")
DEFAULT_ORCA2MKL = Path("/home/gma/bzhang/orca6.0/orca_6_0_0_shared_openmpi416/orca_2mkl")

FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
SURFACE_DIRECT_COLUMNS = {
    "VS_max": ["phys_esp_vs_max_kcal", "phys_esp_global_max_au"],
    "Sigma2_tot": ["phys_esp_sigma2_tot_au2", "phys_esp_sigma2_tot_kcal2", "phys_esp_sigma2_nu_au2"],
    "Nu": ["phys_esp_nu", "phys_esp_sigma2_nu_au2", "phys_esp_sigma2_nu_kcal2"],
    "Trigger_Bond_Rho": ["phys_qtaim_trigger_bcp_rho"],
}


def fnum(pattern: str, text: str, flags: int = re.I | re.S) -> float:
    m = re.search(pattern, text, flags)
    if not m:
        return math.nan
    try:
        return float(m.group(1))
    except Exception:
        return math.nan


def inum(pattern: str, text: str, flags: int = re.I | re.S) -> float:
    m = re.search(pattern, text, flags)
    if not m:
        return math.nan
    try:
        return int(float(m.group(1)))
    except Exception:
        return math.nan


def finite_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def locate_wavefunction(row: pd.Series, active_root: Path) -> Tuple[str, str, str]:
    mol = str(row.get("Molecule", "")).strip()
    if not mol:
        return "", "", "missing_molecule"

    active_dir = active_root / "temp_calc" / mol
    active_molden = active_dir / f"{mol}_step2_freq.molden.input"
    if active_molden.exists() and active_molden.stat().st_size > 1000:
        return str(active_molden), "", "active_step2_molden"

    active_gbw = active_dir / f"{mol}_step2_freq.gbw"
    if active_gbw.exists() and active_gbw.stat().st_size > 1000:
        return "", str(active_gbw), "active_step2_gbw"

    active_bde_parent = active_root / "orca_bde_full_library" / "live_jobs" / mol / "parent.gbw"
    if active_bde_parent.exists() and active_bde_parent.stat().st_size > 1000:
        return "", str(active_bde_parent), "active_bde_parent_gbw"

    jobdir = str(row.get("BDE_Job_Dir", "")).strip()
    if jobdir and jobdir.lower() != "nan":
        candidates = [Path(jobdir)]
        if "/jobs/" in jobdir:
            candidates.append(Path(jobdir.replace("/jobs/", "/live_jobs/")))
        if "/live_jobs/" in jobdir:
            candidates.append(Path(jobdir.replace("/live_jobs/", "/jobs/")))
        for d in candidates:
            parent = d / "parent.gbw"
            if parent.exists() and parent.stat().st_size > 1000:
                return "", str(parent), "bde_parent_gbw"

    return "", "", "missing_wavefunction"


def prepare_manifest(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.dataset)
    active_bde: Dict[str, Dict[str, object]] = {}
    active_final = Path(args.active_root) / "data" / "final_verification_results.csv"
    if active_final.exists():
        active_df = pd.read_csv(active_final)
        keep_cols = [
            c
            for c in [
                "Molecule",
                "BDE_Bond_Type",
                "BDE_Bond_i_1based",
                "BDE_Bond_j_1based",
                "BDE_Bond_WBO",
                "BDE_Parse_Status",
                "BDE_Job_Dir",
            ]
            if c in active_df.columns
        ]
        active_bde = active_df[keep_cols].drop_duplicates("Molecule", keep="last").set_index("Molecule").to_dict("index")
    rows = []
    for idx, row in df.iterrows():
        source_molden, source_gbw, source_kind = locate_wavefunction(row, Path(args.active_root))
        mol = row.get("Molecule", "")
        bde_info = active_bde.get(str(mol), {})
        bond_type = row.get("BDE_Bond_Type", "")
        bond_i = row.get("BDE_Bond_i_1based", "")
        bond_j = row.get("BDE_Bond_j_1based", "")
        if (pd.isna(bond_i) or pd.isna(bond_j) or bond_i == "" or bond_j == "") and bde_info:
            bond_type = bde_info.get("BDE_Bond_Type", bond_type)
            bond_i = bde_info.get("BDE_Bond_i_1based", bond_i)
            bond_j = bde_info.get("BDE_Bond_j_1based", bond_j)
        rows.append(
            {
                "row_index": idx,
                "Molecule": mol,
                "SMILES": row.get("SMILES", ""),
                "Source_Group": row.get("Source_Group", ""),
                "BDE_Bond_Type": bond_type,
                "BDE_Bond_i_1based": bond_i,
                "BDE_Bond_j_1based": bond_j,
                "source_molden": source_molden,
                "source_gbw": source_gbw,
                "source_kind": source_kind,
            }
        )
    man = pd.DataFrame(rows)
    manifest = Path(args.manifest) if args.manifest else out_root / "true_phys_feature_manifest_20260624.csv"
    man.to_csv(manifest, index=False)
    qa = man.groupby(["Source_Group", "source_kind"]).size().reset_index(name="n")
    qa_path = out_root / "true_phys_feature_manifest_qa_20260624.csv"
    qa.to_csv(qa_path, index=False)
    print(f"WROTE {manifest}")
    print(f"WROTE {qa_path}")
    print(qa.to_string(index=False))


def run_command(cmd: List[str], cwd: Path, input_text: Optional[str], timeout: int, env: Dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
    )


def prepare_molden(row: pd.Series, workdir: Path, orca2mkl: Path) -> Tuple[Optional[Path], str]:
    workdir.mkdir(parents=True, exist_ok=True)
    source_molden = str(row.get("source_molden", "")).strip()
    source_gbw = str(row.get("source_gbw", "")).strip()
    mol = str(row.get("Molecule", "molecule")).strip() or "molecule"

    if source_molden and Path(source_molden).exists():
        dst = workdir / f"{mol}.molden.input"
        safe_link_or_copy(Path(source_molden), dst)
        return dst, "molden_ready"

    if source_gbw and Path(source_gbw).exists():
        gbw_dst = workdir / "parent.gbw"
        safe_link_or_copy(Path(source_gbw), gbw_dst)
        molden = workdir / "parent.molden.input"
        if molden.exists() and molden.stat().st_size > 1000:
            return molden, "molden_ready"
        if not orca2mkl.exists():
            return None, "missing_orca_2mkl"
        log = workdir / "orca_2mkl.log"
        res = run_command([str(orca2mkl), "parent", "-molden"], workdir, None, 180, os.environ.copy())
        log.write_text((res.stdout or "") + ("\n=== STDERR ===\n" + res.stderr if res.stderr else ""), errors="ignore")
        if molden.exists() and molden.stat().st_size > 1000:
            return molden, "molden_ready"
        return None, f"orca_2mkl_failed_rc{res.returncode}"

    return None, "missing_wavefunction"


def parse_esp(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["phys_esp_global_min_au"] = fnum(r"Global surface minimum:\s*(%s)\s*a\.u\." % FLOAT_RE, text)
    out["phys_esp_global_max_au"] = fnum(r"Global surface maximum:\s*(%s)\s*a\.u\." % FLOAT_RE, text)
    out["phys_esp_surface_minima_count"] = inum(r"The number of surface minima:\s*(\d+)", text)
    out["phys_esp_surface_maxima_count"] = inum(r"The number of surface maxima:\s*(\d+)", text)
    out["phys_esp_volume_bohr3"] = fnum(r"Volume:\s*(%s)\s*Bohr\^3" % FLOAT_RE, text)
    out["phys_esp_volume_ang3"] = fnum(r"Volume:\s*%s\s*Bohr\^3\s*\(\s*(%s)\s*Angstrom\^3" % (FLOAT_RE, FLOAT_RE), text)
    out["phys_esp_est_density_gcm3"] = fnum(r"Estimated density according to mass and volume.*?:\s*(%s)\s*g/cm\^3" % FLOAT_RE, text)
    out["phys_esp_vs_min_kcal"] = fnum(r"Minimal value:\s*(%s)\s*kcal/mol" % FLOAT_RE, text)
    out["phys_esp_vs_max_kcal"] = fnum(r"Maximal value:\s*(%s)\s*kcal/mol" % FLOAT_RE, text)

    for label, prefix in [
        ("Overall surface area", "phys_esp_area_total"),
        ("Positive surface area", "phys_esp_area_positive"),
        ("Negative surface area", "phys_esp_area_negative"),
    ]:
        out[f"{prefix}_bohr2"] = fnum(r"%s:\s*(%s)\s*Bohr\^2" % (re.escape(label), FLOAT_RE), text)
        out[f"{prefix}_ang2"] = fnum(r"%s:\s*%s\s*Bohr\^2\s*\(\s*(%s)\s*Angstrom\^2" % (re.escape(label), FLOAT_RE, FLOAT_RE), text)

    for label, prefix in [
        ("Overall average value", "phys_esp_avg_total"),
        ("Positive average value", "phys_esp_avg_positive"),
        ("Negative average value", "phys_esp_avg_negative"),
    ]:
        out[f"{prefix}_au"] = fnum(r"%s:\s*(%s)\s*a\.u\." % (re.escape(label), FLOAT_RE), text)
        out[f"{prefix}_kcal"] = fnum(r"%s:\s*%s\s*a\.u\.\s*\(\s*(%s)\s*kcal/mol" % (re.escape(label), FLOAT_RE, FLOAT_RE), text)

    for label, prefix in [
        ("Overall variance \\(sigma\\^2_tot\\)", "phys_esp_sigma2_tot"),
        ("Positive variance", "phys_esp_variance_positive"),
        ("Negative variance", "phys_esp_variance_negative"),
    ]:
        out[f"{prefix}_au2"] = fnum(r"%s:\s*(%s)\s*a\.u\.\^2" % (label, FLOAT_RE), text)
        out[f"{prefix}_kcal2"] = fnum(r"%s:\s*%s\s*a\.u\.\^2\s*\(\s*(%s)" % (label, FLOAT_RE, FLOAT_RE), text)

    out["phys_esp_nu"] = fnum(r"Balance of charges \(nu\):\s*(%s)" % FLOAT_RE, text)
    out["phys_esp_sigma2_nu_au2"] = fnum(r"Product of sigma\^2_tot and nu:\s*(%s)\s*a\.u\.\^2" % FLOAT_RE, text)
    out["phys_esp_sigma2_nu_kcal2"] = fnum(r"Product of sigma\^2_tot and nu:\s*%s\s*a\.u\.\^2\s*\(\s*(%s)" % (FLOAT_RE, FLOAT_RE), text)
    out["phys_esp_pi_au"] = fnum(r"Internal charge separation \(Pi\):\s*(%s)\s*a\.u\." % FLOAT_RE, text)
    out["phys_esp_pi_kcal"] = fnum(r"Internal charge separation \(Pi\):\s*%s\s*a\.u\.\s*\(\s*(%s)\s*kcal/mol" % (FLOAT_RE, FLOAT_RE), text)
    out["phys_esp_mpi_ev"] = fnum(r"Molecular polarity index \(MPI\):\s*(%s)\s*eV" % FLOAT_RE, text)
    out["phys_esp_mpi_kcal"] = fnum(r"Molecular polarity index \(MPI\):\s*%s\s*eV\s*\(\s*(%s)\s*kcal/mol" % (FLOAT_RE, FLOAT_RE), text)
    out["phys_esp_area_nonpolar_ang2"] = fnum(r"Nonpolar surface area.*?:\s*(%s)\s*Angstrom\^2" % FLOAT_RE, text)
    out["phys_esp_area_nonpolar_pct"] = fnum(r"Nonpolar surface area.*?\(\s*(%s)\s*%%" % FLOAT_RE, text)
    out["phys_esp_area_polar_ang2"] = fnum(r"Polar surface area.*?:\s*(%s)\s*Angstrom\^2" % FLOAT_RE, text)
    out["phys_esp_area_polar_pct"] = fnum(r"Polar surface area.*?\(\s*(%s)\s*%%" % FLOAT_RE, text)
    out["phys_esp_skew_total"] = fnum(r"Overall skewness:\s*(%s)" % FLOAT_RE, text)
    out["phys_esp_skew_positive"] = fnum(r"Positive skewness:\s*(%s)" % FLOAT_RE, text)
    out["phys_esp_skew_negative"] = fnum(r"Negative skewness:\s*(%s)" % FLOAT_RE, text)
    return out


def parse_cp_value(block: str, label: str) -> float:
    return fnum(re.escape(label) + r":\s*(%s)" % FLOAT_RE, block)


def parse_cp_props(path: Path, trigger_i: Optional[int], trigger_j: Optional[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not path.exists() or path.stat().st_size == 0:
        out["phys_qtaim_status_ok"] = 0
        return out

    text = path.read_text(errors="ignore")
    header = re.compile(r"-+\s+CP\s+(\d+),\s+Type\s+\(3,-1\)\s+-+")
    matches = list(header.finditer(text))
    bcps: List[Dict[str, float]] = []
    for i, m in enumerate(matches):
        block = text[m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        conn = re.search(
            r"Connected atoms:\s*(\d+)\(([A-Za-z]+)\s*\)\s*--\s*(\d+)\(([A-Za-z]+)\s*\)",
            block,
        )
        if not conn:
            continue
        atom_i, sym_i, atom_j, sym_j = int(conn.group(1)), conn.group(2), int(conn.group(3)), conn.group(4)
        pair = "_".join(sorted([sym_i, sym_j]))
        item = {
            "cp_index": float(m.group(1)),
            "atom_i": float(atom_i),
            "atom_j": float(atom_j),
            "pair": pair,
            "rho": parse_cp_value(block, "Density of all electrons"),
            "laplacian": parse_cp_value(block, "Laplacian of electron density"),
            "G": parse_cp_value(block, "Lagrangian kinetic energy G(r)"),
            "K": parse_cp_value(block, "Hamiltonian kinetic energy K(r)"),
            "V": parse_cp_value(block, "Potential energy density V(r)"),
            "H": parse_cp_value(block, "Energy density E(r) or H(r)"),
            "ELF": parse_cp_value(block, "Electron localization function (ELF)"),
            "LOL": parse_cp_value(block, "Localized orbital locator (LOL)"),
            "IRI": parse_cp_value(block, "Interaction region indicator (IRI)"),
            "RDG": parse_cp_value(block, "Reduced density gradient (RDG)"),
            "sign_lambda2_rho": parse_cp_value(block, "Sign(lambda2)*rho"),
            "ALIE": parse_cp_value(block, "Average local ionization energy (ALIE)"),
            "total_esp_au": fnum(r"Total ESP:\s*(%s)\s*a\.u\." % FLOAT_RE, block),
            "ellipticity": parse_cp_value(block, "Ellipticity of electron density"),
            "eta": parse_cp_value(block, "eta index"),
            "stiffness": parse_cp_value(block, "Stiffness"),
            "stress_stiffness": parse_cp_value(block, "Stress tensor stiffness"),
            "stress_polarizability": parse_cp_value(block, "Stress tensor polarizability"),
        }
        item["is_trigger"] = float(
            trigger_i is not None
            and trigger_j is not None
            and {atom_i, atom_j} == {trigger_i, trigger_j}
        )
        bcps.append(item)

    out["phys_qtaim_status_ok"] = 1 if bcps else 0
    out["phys_qtaim_bcp_count"] = float(len(bcps))
    if not bcps:
        return out

    def emit_stats(prefix: str, items: List[Dict[str, float]], fields: Iterable[str]) -> None:
        out[f"{prefix}_count"] = float(len(items))
        for field in fields:
            vals = np.array([x[field] for x in items if np.isfinite(x.get(field, math.nan))], dtype=float)
            if vals.size == 0:
                for suffix in ["min", "mean", "std", "p05", "p50", "p95", "max"]:
                    out[f"{prefix}_{field}_{suffix}"] = math.nan
                continue
            out[f"{prefix}_{field}_min"] = float(np.min(vals))
            out[f"{prefix}_{field}_mean"] = float(np.mean(vals))
            out[f"{prefix}_{field}_std"] = float(np.std(vals))
            out[f"{prefix}_{field}_p05"] = float(np.percentile(vals, 5))
            out[f"{prefix}_{field}_p50"] = float(np.percentile(vals, 50))
            out[f"{prefix}_{field}_p95"] = float(np.percentile(vals, 95))
            out[f"{prefix}_{field}_max"] = float(np.max(vals))

    stat_fields = ["rho", "laplacian", "H", "V", "G", "ELF", "LOL", "ellipticity", "sign_lambda2_rho"]
    emit_stats("phys_qtaim_all_bcp", bcps, stat_fields)

    trigger = [x for x in bcps if x["is_trigger"] == 1.0]
    out["phys_qtaim_trigger_found"] = 1.0 if trigger else 0.0
    if trigger:
        t = trigger[0]
        for field in stat_fields + ["K", "IRI", "RDG", "ALIE", "total_esp_au", "eta", "stiffness", "stress_stiffness", "stress_polarizability"]:
            out[f"phys_qtaim_trigger_bcp_{field}"] = float(t.get(field, math.nan))
    else:
        for field in stat_fields + ["K", "IRI", "RDG", "ALIE", "total_esp_au", "eta", "stiffness", "stress_stiffness", "stress_polarizability"]:
            out[f"phys_qtaim_trigger_bcp_{field}"] = math.nan

    non_trigger = [x for x in bcps if x["is_trigger"] != 1.0]
    emit_stats("phys_qtaim_nontrigger_bcp", non_trigger, stat_fields)

    for pair in ["C_N", "N_N", "N_O", "C_O", "C_C", "H_N", "H_C", "H_O", "O_O"]:
        pair_items = [x for x in bcps if x["pair"] == pair]
        emit_stats(f"phys_qtaim_pair_{pair}", pair_items, ["rho", "laplacian", "H", "ellipticity"])

    return out


def extract_one(row: pd.Series, args: argparse.Namespace) -> Dict[str, object]:
    mol = str(row.get("Molecule", "")).strip()
    out_root = Path(args.out_root)
    workdir = out_root / "work" / mol
    record: Dict[str, object] = {
        "row_index": row.get("row_index", ""),
        "Molecule": mol,
        "SMILES": row.get("SMILES", ""),
        "Source_Group": row.get("Source_Group", ""),
        "source_kind": row.get("source_kind", ""),
    }
    molden, molden_status = prepare_molden(row, workdir, Path(args.orca2mkl))
    record["molden_status"] = molden_status
    if molden is None:
        record["extract_status"] = "missing_molden"
        return record

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    env["OMP_STACKSIZE"] = "2G"

    esp_out = workdir / "esp_output.txt"
    if not esp_out.exists() or esp_out.stat().st_size < 500:
        try:
            res = run_command(
                [str(args.multiwfn), str(molden)],
                workdir,
                "12\n0\n-1\n-1\nq\n",
                args.esp_timeout,
                env,
            )
            esp_out.write_text(res.stdout or "", errors="ignore")
            if res.stderr:
                (workdir / "esp.stderr.txt").write_text(res.stderr, errors="ignore")
        except subprocess.TimeoutExpired:
            record["esp_status"] = "timeout"
        except Exception as exc:
            record["esp_status"] = f"failed:{type(exc).__name__}"
    if esp_out.exists():
        esp_text = esp_out.read_text(errors="ignore")
        record.update(parse_esp(esp_text))
        record["esp_status"] = "ok" if "Summary of surface analysis" in esp_text else record.get("esp_status", "parse_missing_summary")

    cp_path = workdir / "CPprop.txt"
    if not cp_path.exists() or cp_path.stat().st_size < 500:
        try:
            res = run_command(
                [str(args.multiwfn), str(molden)],
                workdir,
                "2\n2\n3\n8\n7\n0\n-10\nq\n",
                args.qtaim_timeout,
                env,
            )
            (workdir / "topology_output.txt").write_text(res.stdout or "", errors="ignore")
            if res.stderr:
                (workdir / "topology.stderr.txt").write_text(res.stderr, errors="ignore")
        except subprocess.TimeoutExpired:
            record["qtaim_status"] = "timeout"
        except Exception as exc:
            record["qtaim_status"] = f"failed:{type(exc).__name__}"

    trigger_i = finite_int(row.get("BDE_Bond_i_1based"))
    trigger_j = finite_int(row.get("BDE_Bond_j_1based"))
    qtaim = parse_cp_props(cp_path, trigger_i, trigger_j)
    record.update(qtaim)
    if "qtaim_status" not in record:
        record["qtaim_status"] = "ok" if qtaim.get("phys_qtaim_status_ok", 0) == 1 else "parse_no_bcp"
    record["extract_status"] = "ok" if record.get("esp_status") == "ok" and record.get("qtaim_status") == "ok" else "partial"
    return record


def extract(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    parts_dir = out_root / "features_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(args.manifest)
    if args.limit is not None:
        man = man.head(args.limit).copy()
    if args.task_index is not None:
        start = int(args.task_index) * int(args.chunk_size)
        end = min(start + int(args.chunk_size), len(man))
        sub = man.iloc[start:end].copy()
        part_name = f"true_phys_features_part_{int(args.task_index):04d}.csv"
    else:
        sub = man.copy()
        part_name = "true_phys_features_part_manual.csv"
    print(f"EXTRACT rows={len(sub)} part={part_name}")
    rows = []
    for n, (_, row) in enumerate(sub.iterrows(), 1):
        try:
            rec = extract_one(row, args)
        except Exception as exc:
            rec = {
                "row_index": row.get("row_index", ""),
                "Molecule": row.get("Molecule", ""),
                "SMILES": row.get("SMILES", ""),
                "Source_Group": row.get("Source_Group", ""),
                "source_kind": row.get("source_kind", ""),
                "extract_status": f"exception:{type(exc).__name__}",
                "error_message": str(exc)[:500],
            }
        rows.append(rec)
        print(f"[{n}/{len(sub)}] {rec.get('Molecule')} {rec.get('extract_status')} esp={rec.get('esp_status')} qtaim={rec.get('qtaim_status')}")
    out = pd.DataFrame(rows)
    out_path = parts_dir / part_name
    out.to_csv(out_path, index=False)
    print(f"WROTE {out_path}")


def summarize(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    parts = sorted((out_root / "features_parts").glob("true_phys_features_part_*.csv"))
    if not parts:
        raise SystemExit("No feature parts found")
    frames = [pd.read_csv(p) for p in parts if p.stat().st_size > 0]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["row_index", "Molecule"]).drop_duplicates(subset=["row_index", "Molecule"], keep="last")
    merged = out_root / "true_phys_features_merged_20260624.csv"
    df.to_csv(merged, index=False)
    status = (
        df.groupby(["extract_status", "esp_status", "qtaim_status"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    status_path = out_root / "true_phys_features_status_20260624.csv"
    status.to_csv(status_path, index=False)
    numeric_cols = [c for c in df.columns if c.startswith("phys_")]
    coverage = []
    for c in numeric_cols:
        coverage.append({"feature": c, "nonnull": int(df[c].notna().sum()), "n": len(df)})
    coverage_df = pd.DataFrame(coverage).sort_values(["nonnull", "feature"], ascending=[False, True])
    coverage_path = out_root / "true_phys_features_coverage_20260624.csv"
    coverage_df.to_csv(coverage_path, index=False)
    print(f"WROTE {merged}")
    print(f"WROTE {status_path}")
    print(f"WROTE {coverage_path}")
    print(status.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="True surface-ESP/QTAIM feature extraction")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-manifest")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--active-root", default=str(DEFAULT_ACTIVE_ROOT))
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--manifest", default="")
    p.set_defaults(func=prepare_manifest)

    p = sub.add_parser("extract")
    p.add_argument("--manifest", default=str(DEFAULT_OUT_ROOT / "true_phys_feature_manifest_20260624.csv"))
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--multiwfn", default=str(DEFAULT_MULTIWFN))
    p.add_argument("--orca2mkl", default=str(DEFAULT_ORCA2MKL))
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--esp-timeout", type=int, default=900)
    p.add_argument("--qtaim-timeout", type=int, default=900)
    p.add_argument("--task-index", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=25)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=extract)

    p = sub.add_parser("summarize")
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.set_defaults(func=summarize)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
