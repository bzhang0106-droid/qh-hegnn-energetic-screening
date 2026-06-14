from pathlib import Path

import pandas as pd

ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
TRUE = ROOT / "results" / "True_vs_Pred_Detonation.csv"


def main() -> None:
    true = pd.read_csv(TRUE)
    missing = []
    for mol in true["Molecule"].astype(str):
        d = ROOT / "temp_calc" / mol
        if not (d / "esp_output.txt").exists() or not (d / "critic2_cpreport.out").exists():
            missing.append(mol)
    print(f"08a AL08 readiness check: rows={len(true)}, missing={len(missing)}")
    if missing:
        print("missing_first20=" + ",".join(missing[:20]))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
