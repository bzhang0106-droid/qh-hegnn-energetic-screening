import os
import re
import math
import argparse
import pandas as pd
import numpy as np

NA_FACTOR = 0.602214076  # 工业基准分子库：用于 density proxy -> crystal density calibration
# D_ref / P_ref 对新补 benchmark 暂不强制使用；density calibration 主要依赖 Crystal_Density_ref。
BENCHMARKS = {
    "TNT": {
        "formula": "c7h5n3o6",
        "C": 7, "H": 5, "N": 3, "O": 6,
        "MW": 227.13110,
        "D_ref(m/s)": np.nan,
        "P_ref(GPa)": np.nan,
        "Crystal_Density_ref(g/cm3)": 1.654,
        "Reference_Source": "TNT density 1.654 g/cm3; PubChem/NTP/Merck Index.",
    },
    "RDX": {
        "formula": "c3h6n6o6",
        "C": 3, "H": 6, "N": 6, "O": 6,
        "MW": 222.03488193738,
        "D_ref(m/s)": 8750.0,
        "P_ref(GPa)": 34.0,
        "Crystal_Density_ref(g/cm3)": 1.81,
        "Reference_Source": "Existing benchmark reference.",
    },
    "HMX": {
        "formula": "c4h8n8o8",
        "C": 4, "H": 8, "N": 8, "O": 8,
        "MW": 296.04650924984,
        "D_ref(m/s)": 9100.0,
        "P_ref(GPa)": 39.0,
        "Crystal_Density_ref(g/cm3)": 1.90,
        "Reference_Source": "Existing benchmark reference.",
    },
    "epsilon-CL-20": {
        "formula": "c6h6n12o12",
        "C": 6, "H": 6, "N": 12, "O": 12,
        "MW": 438.02281368138,
        "D_ref(m/s)": 9650.0,
        "P_ref(GPa)": 44.0,
        "Crystal_Density_ref(g/cm3)": 2.04,
        "Reference_Source": "Existing benchmark reference.",
    },
    "PETN": {
        "formula": "c5h8n4o12",
        "C": 5, "H": 8, "N": 4, "O": 12,
        "MW": 316.01387171040,
        "D_ref(m/s)": 8400.0,
        "P_ref(GPa)": 33.0,
        "Crystal_Density_ref(g/cm3)": 1.77,
        "Reference_Source": "Existing benchmark reference.",
    },
    "TATB": {
        "formula": "c6h6n6o6",
        "C": 6, "H": 6, "N": 6, "O": 6,
        "MW": 258.14800,
        "D_ref(m/s)": np.nan,
        "P_ref(GPa)": np.nan,
        "Crystal_Density_ref(g/cm3)": 1.937,
        "Reference_Source": "TATB theoretical maximum crystal density 1.937 g/cm3.",
    },
    "FOX-7": {
        "formula": "c2h4n4o4",
        "C": 2, "H": 4, "N": 4, "O": 4,
        "MW": 148.07860,
        "D_ref(m/s)": np.nan,
        "P_ref(GPa)": np.nan,
        "Crystal_Density_ref(g/cm3)": 1.885,
        "Reference_Source": "FOX-7 reported density 1.885 g/cm3; verify final citation before manuscript.",
    },
    "NTO": {
        "formula": "c2h2n4o3",
        "C": 2, "H": 2, "N": 4, "O": 3,
        "MW": 130.06330,
        "D_ref(m/s)": np.nan,
        "P_ref(GPa)": np.nan,
        "Crystal_Density_ref(g/cm3)": 1.93,
        "Reference_Source": "NTO density literature value; EPA CompTox reports experimental range around 1.90-1.93 g/cm3.",
    },
}


def parse_esp_output(path):
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    out = {}

    # Multiwfn 常见体积输出
    patterns = [
        (r"Volume enclosed by the isosurface:.*?\(?\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*Angstrom\^3", "A3"),
        (r"Molecular volume:\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*Angstrom", "A3"),
        (r"Volume:\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*Angstrom\^3", "A3"),
        (r"Volume enclosed by the isosurface:.*?\(?\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*Bohr\^3", "Bohr3"),
    ]
    for pat, unit in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            val = float(m.group(1))
            if unit == "Bohr3":
                val = val * (0.529177210903 ** 3)
            out["Volume_A3"] = val
            break

    m = re.search(r"Maximal value:\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*kcal/mol", text, re.IGNORECASE)
    if m:
        out["VS_max"] = float(m.group(1))
    else:
        m = re.search(r"Global surface maximum:\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*a\.u\.", text, re.IGNORECASE)
        if m:
            out["VS_max"] = float(m.group(1)) * 627.509

    m = re.search(r"(?:Overall variance|Variance of ESP)[^:]*:\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", text, re.IGNORECASE)
    if m:
        out["Sigma2_tot"] = float(m.group(1))

    m = re.search(r"Balance of charges[^:]*:\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", text, re.IGNORECASE)
    if m:
        out["Nu"] = float(m.group(1))

    return out

def parse_critic2_rho(path):
    if not os.path.exists(path):
        return np.nan

    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "(3,-1)" in line and "bond" in line.lower():
                parts = line.split()
                # 兼容你原脚本使用 parts[7] 的格式
                for idx in [7, 6, 8]:
                    if len(parts) > idx:
                        try:
                            rho = float(parts[idx])
                            if 0.02 < rho < 1.0:
                                vals.append(rho)
                                break
                        except Exception:
                            pass
    return min(vals) if vals else np.nan

def kj_calc(C, H, N, O, MW, HOF, rho):
    if not all(np.isfinite(x) for x in [MW, HOF, rho]) or MW <= 0 or rho <= 0:
        return np.nan, np.nan

    o_avail = O
    h2o = min(H / 2.0, o_avail)
    o_avail -= h2o
    h2 = H / 2.0 - h2o

    co = min(C, o_avail)
    o_avail -= co

    co2 = min(co, o_avail)
    o_avail -= co2
    co -= co2

    o2 = o_avail / 2.0
    n2 = N / 2.0

    total_gas = h2o + h2 + co + co2 + o2 + n2
    if total_gas <= 0:
        return np.nan, np.nan

    gas_mass = h2o*18.015 + h2*2.016 + co*28.01 + co2*44.01 + o2*31.998 + n2*28.013
    N_gas = total_gas / MW
    M_gas = gas_mass / total_gas

    hof_products = h2o*(-57.8) + co*(-26.4) + co2*(-94.0)
    Q = (HOF - hof_products) / MW * 1000.0
    if Q <= 0:
        return np.nan, np.nan

    D = 1.01 * (N_gas * (M_gas**0.5) * (Q**0.5))**0.5 * (1.0 + 1.30*rho) * 1000.0
    P = 1.558 * (rho**2) * N_gas * (M_gas**0.5) * (Q**0.5)
    return D, P

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents_root", default="../../B3LYP-D3BJ")
    parser.add_argument("--output", default="../data/industrial_benchmarks_v2.csv")
    parser.add_argument("--calibration_output", default="../data/density_calibration_reference.csv")
    args = parser.parse_args()

    root = os.path.abspath(args.parents_root)

    rows = []
    for name, meta in BENCHMARKS.items():
        formula = meta["formula"]
        d = os.path.join(root, "parents_step3_dlpnomp2", formula)
        esp = os.path.join(d, "esp_output.txt")
        critic = os.path.join(d, "critic2_cpreport.out")

        esp_data = parse_esp_output(esp)
        rho_bcp = parse_critic2_rho(critic)

        vol = esp_data.get("Volume_A3", np.nan)
        density_proxy = meta["MW"] / (NA_FACTOR * vol) if np.isfinite(vol) and vol > 0 else np.nan

        rows.append({
            "Molecule": name,
            "Formula": formula,
            "C": meta["C"],
            "H": meta["H"],
            "N": meta["N"],
            "O": meta["O"],
            "MW": meta["MW"],
            "Volume_A3_from_Multiwfn": vol,
            "Density_proxy(g/cm3)": density_proxy,
            "Crystal_Density_ref(g/cm3)": meta["Crystal_Density_ref(g/cm3)"],
            "D_ref(m/s)": meta["D_ref(m/s)"],
            "P_ref(GPa)": meta["P_ref(GPa)"],
            "Trigger_Bond_Rho": rho_bcp,
            "VS_max": esp_data.get("VS_max", np.nan),
            "Sigma2_tot": esp_data.get("Sigma2_tot", np.nan),
            "Nu": esp_data.get("Nu", np.nan),
            "ESP_Source": esp if os.path.exists(esp) else "",
            "QTAIM_Source": critic if os.path.exists(critic) else "",
            "Reference_Source": meta.get("Reference_Source", ""),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    calib = df[["Molecule", "Density_proxy(g/cm3)", "Crystal_Density_ref(g/cm3)"]].copy()
    calib.rename(columns={
        "Density_proxy(g/cm3)": "Density_proxy",
        "Crystal_Density_ref(g/cm3)": "Crystal_Density(g/cm3)"
    }, inplace=True)
    calib["Source"] = "industrial_benchmark_reference"
    ref_map = df.set_index("Molecule")["Reference_Source"].to_dict() if "Reference_Source" in df.columns else {}
    calib["Reference_Source"] = calib["Molecule"].map(ref_map).fillna("")
    calib.to_csv(args.calibration_output, index=False)

    print("=" * 100)
    print(f"[OK] Saved benchmark table: {args.output}")
    print(f"[OK] Saved density calibration reference: {args.calibration_output}")
    print("=" * 100)
    print(df.to_string(index=False))

    if df["Volume_A3_from_Multiwfn"].isna().any():
        print("\n[WARN] Some benchmark molecules have no parsed Multiwfn volume. Check esp_output.txt format.")
    if df["Trigger_Bond_Rho"].isna().any():
        print("\n[WARN] Some benchmark molecules have no parsed QTAIM rho. Check critic2_cpreport.out.")

if __name__ == "__main__":
    main()
