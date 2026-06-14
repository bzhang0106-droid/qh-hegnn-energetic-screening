import re
from pathlib import Path
import numpy as np
import pandas as pd

MANIFEST = Path("../data/electronic_features/xtb_tasks.tsv")
OUT = Path("../data/electronic_features/xtb_features.csv")

COMMON_PAIR_KEYS = [
    "C_C", "C_H", "C_N", "C_O",
    "H_N", "H_O",
    "N_N", "N_O",
    "O_O",
]

def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""

def parse_float_regex(patterns, text):
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    return np.nan

def all_floats(line: str):
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+(?:[Ee][-+]?\d+)", line)]

def parse_total_energy(text: str):
    return parse_float_regex([
        r"\|\s*TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh",
        r"::\s*total energy\s+(-?\d+\.\d+)\s+Eh",
        r"TOTAL ENERGY\s+(-?\d+\.\d+)",
    ], text)

def parse_gap(text: str):
    return parse_float_regex([
        r"\|\s*HOMO-LUMO GAP\s+(-?\d+\.\d+)\s+eV",
        r"::\s*HOMO-LUMO gap\s+(-?\d+\.\d+)\s+eV",
        r"HL-Gap\s+[-+]?\d*\.\d+\s+Eh\s+(-?\d+\.\d+)\s+eV",
        r"HOMO-LUMO GAP\s+(-?\d+\.\d+)\s+eV",
    ], text)

def parse_homo_lumo_ev(text: str):
    homo_vals = []
    lumo_vals = []

    for line in text.splitlines():
        if "(HOMO)" in line:
            vals = all_floats(line)
            # Example:
            # 40 1.9997 -0.3647563 -9.9255 (HOMO)
            # Last numeric value is usually eV orbital energy.
            if vals:
                homo_vals.append(vals[-1])
        elif "(LUMO)" in line:
            vals = all_floats(line)
            if vals:
                lumo_vals.append(vals[-1])

    homo = homo_vals[-1] if homo_vals else np.nan
    lumo = lumo_vals[-1] if lumo_vals else np.nan
    return homo, lumo

def parse_dipole(text: str):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "molecular dipole" in line.lower():
            block = lines[i:i+25]
            # Prefer a line containing "full" or "total"
            for b in block:
                bl = b.lower()
                if "full" in bl or "total" in bl:
                    vals = all_floats(b)
                    if vals:
                        return vals[-1]
            # Otherwise use the last line in block with >=3 floats
            candidates = []
            for b in block:
                vals = all_floats(b)
                if len(vals) >= 3:
                    candidates.append(vals[-1])
            if candidates:
                return candidates[-1]
    return np.nan

def parse_xyz_elements(xyz_path: Path):
    elems = []
    if not xyz_path.exists():
        return elems

    lines = xyz_path.read_text(errors="ignore").splitlines()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            elems.append(parts[0])
    return elems

def parse_charges(path: Path, elems):
    vals = []
    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            fs = all_floats(line)
            if fs:
                vals.append(fs[-1])

    arr = np.array(vals, dtype=float)
    rec = {}

    if arr.size == 0:
        base_keys = [
            "n", "mean", "std", "min", "max", "absmean", "absmax",
            "positive_sum", "negative_sum"
        ]
        for k in base_keys:
            rec[f"xtb_charge_{k}"] = np.nan if k != "n" else 0
    else:
        rec.update({
            "xtb_charge_n": int(arr.size),
            "xtb_charge_mean": float(np.mean(arr)),
            "xtb_charge_std": float(np.std(arr)),
            "xtb_charge_min": float(np.min(arr)),
            "xtb_charge_max": float(np.max(arr)),
            "xtb_charge_absmean": float(np.mean(np.abs(arr))),
            "xtb_charge_absmax": float(np.max(np.abs(arr))),
            "xtb_charge_positive_sum": float(np.sum(arr[arr > 0])) if np.any(arr > 0) else 0.0,
            "xtb_charge_negative_sum": float(np.sum(arr[arr < 0])) if np.any(arr < 0) else 0.0,
        })

    # Element-wise charge statistics for energetic elements.
    for el in ["C", "H", "N", "O", "F", "Cl"]:
        idx = [i for i, e in enumerate(elems) if e == el and i < arr.size]
        sub = arr[idx] if idx and arr.size else np.array([], dtype=float)

        rec[f"xtb_charge_{el}_count"] = int(sub.size)
        rec[f"xtb_charge_{el}_mean"] = float(np.mean(sub)) if sub.size else np.nan
        rec[f"xtb_charge_{el}_min"] = float(np.min(sub)) if sub.size else np.nan
        rec[f"xtb_charge_{el}_max"] = float(np.max(sub)) if sub.size else np.nan
        rec[f"xtb_charge_{el}_absmax"] = float(np.max(np.abs(sub))) if sub.size else np.nan

    return rec

def parse_wbo(path: Path, elems):
    pairs = []
    vals = []

    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue

            # Typical xTB wbo line: i j value
            try:
                i = int(parts[0]) - 1
                j = int(parts[1]) - 1
                val = float(parts[-1])
            except Exception:
                continue

            if i < 0 or j < 0 or i >= len(elems) or j >= len(elems):
                continue

            if not np.isfinite(val):
                continue

            e1, e2 = elems[i], elems[j]
            key = "_".join(sorted([e1, e2]))
            pairs.append((key, val))
            vals.append(val)

    arr = np.array(vals, dtype=float)
    rec = {}

    if arr.size == 0:
        rec.update({
            "xtb_wbo_n": 0,
            "xtb_wbo_mean": np.nan,
            "xtb_wbo_std": np.nan,
            "xtb_wbo_min": np.nan,
            "xtb_wbo_max": np.nan,
            "xtb_wbo_p05": np.nan,
            "xtb_wbo_p50": np.nan,
            "xtb_wbo_p95": np.nan,
            "xtb_wbo_count_lt_0p8": 0,
            "xtb_wbo_count_lt_1p0": 0,
        })
    else:
        rec.update({
            "xtb_wbo_n": int(arr.size),
            "xtb_wbo_mean": float(np.mean(arr)),
            "xtb_wbo_std": float(np.std(arr)),
            "xtb_wbo_min": float(np.min(arr)),
            "xtb_wbo_max": float(np.max(arr)),
            "xtb_wbo_p05": float(np.quantile(arr, 0.05)),
            "xtb_wbo_p50": float(np.quantile(arr, 0.50)),
            "xtb_wbo_p95": float(np.quantile(arr, 0.95)),
            "xtb_wbo_count_lt_0p8": int(np.sum(arr < 0.8)),
            "xtb_wbo_count_lt_1p0": int(np.sum(arr < 1.0)),
        })

    # Pair-specific WBO statistics.
    by_pair = {}
    for key, val in pairs:
        by_pair.setdefault(key, []).append(val)

    for key in COMMON_PAIR_KEYS:
        sub = np.array(by_pair.get(key, []), dtype=float)
        rec[f"xtb_wbo_{key}_count"] = int(sub.size)
        rec[f"xtb_wbo_{key}_mean"] = float(np.mean(sub)) if sub.size else np.nan
        rec[f"xtb_wbo_{key}_min"] = float(np.min(sub)) if sub.size else np.nan
        rec[f"xtb_wbo_{key}_max"] = float(np.max(sub)) if sub.size else np.nan

    # Trigger-bond proxy candidates:
    # energetic trigger bonds often involve weak C-N, N-N, N-O, O-O paths.
    trigger_vals = []
    for key in ["C_N", "N_N", "N_O", "O_O"]:
        trigger_vals.extend(by_pair.get(key, []))

    tr = np.array(trigger_vals, dtype=float)
    rec["xtb_trigger_wbo_proxy_n"] = int(tr.size)
    rec["xtb_trigger_wbo_proxy_min"] = float(np.min(tr)) if tr.size else np.nan
    rec["xtb_trigger_wbo_proxy_mean"] = float(np.mean(tr)) if tr.size else np.nan
    rec["xtb_trigger_wbo_proxy_p05"] = float(np.quantile(tr, 0.05)) if tr.size else np.nan

    return rec

def parse_one(row):
    workdir = Path(row["workdir"])
    out_path = workdir / "xtb.out"
    err_path = workdir / "xtb.err"
    xyz_path = Path(row["xyz_path"])
    charge_path = workdir / "charges"
    wbo_path = workdir / "wbo"

    text = read_text(out_path)
    err = read_text(err_path)
    all_text = text + "\n" + err

    if not out_path.exists():
        status = "missing"
    elif re.search(r"normal termination|finished run|finished", all_text, flags=re.I):
        status = "ok"
    elif re.search(r"error|fatal|failed", all_text, flags=re.I):
        status = "error"
    else:
        status = "unknown"

    elems = parse_xyz_elements(xyz_path)
    homo_ev, lumo_ev = parse_homo_lumo_ev(text)

    rec = {
        "row_index": int(row["row_index"]),
        "example_id": row["example_id"],
        "Molecule": row["molecule"],
        "xtb_status": status,
        "xtb_total_energy_Eh": parse_total_energy(text),
        "xtb_homo_eV": homo_ev,
        "xtb_lumo_eV": lumo_ev,
        "xtb_gap_eV": parse_gap(text),
        "xtb_gap_from_orbitals_eV": float(lumo_ev - homo_ev) if np.isfinite(homo_ev) and np.isfinite(lumo_ev) else np.nan,
        "xtb_dipole_D": parse_dipole(text),
        "xtb_n_atoms": len(elems),
        "xtb_workdir": str(workdir),
    }

    rec.update(parse_charges(charge_path, elems))
    rec.update(parse_wbo(wbo_path, elems))

    return rec

def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    df = pd.read_csv(MANIFEST, sep="\t")
    rows = [parse_one(r) for _, r in df.iterrows()]
    out = pd.DataFrame(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("[OK] saved:", OUT)
    print("rows:", len(out))
    print()
    print("status counts:")
    print(out["xtb_status"].value_counts(dropna=False).to_string())

    key_cols = [
        "xtb_total_energy_Eh",
        "xtb_homo_eV",
        "xtb_lumo_eV",
        "xtb_gap_eV",
        "xtb_gap_from_orbitals_eV",
        "xtb_dipole_D",
        "xtb_charge_std",
        "xtb_charge_absmax",
        "xtb_wbo_mean",
        "xtb_wbo_min",
        "xtb_trigger_wbo_proxy_min",
    ]

    print()
    print("non-null counts:")
    for c in key_cols:
        if c in out.columns:
            print(f"{c:32s} {out[c].notna().sum()}")

    print()
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
