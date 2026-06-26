#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "03_egnn_painn_train.py"
SPLIT_JSON = Path(os.environ["QH_HEGNN_CONTROLLED_SPLIT_JSON"]).resolve()


def load_training_module():
    spec = importlib.util.spec_from_file_location("qh_hegnn_train03_controlled", BASE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    split = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    core = np.asarray(split["core_idx"], dtype=int)
    calib = np.asarray(split["calib_idx"], dtype=int)
    val = np.asarray(split["val_idx"], dtype=int)

    def controlled_select_split_indices(n, seed, val_fraction, calib_fraction):
        all_idx = np.concatenate([core, calib, val])
        if len(set(map(int, all_idx))) != len(all_idx):
            raise RuntimeError("controlled split has overlapping indices")
        if len(all_idx) != int(n):
            raise RuntimeError(f"controlled split length {len(all_idx)} != aligned examples {n}")
        if int(all_idx.min()) < 0 or int(all_idx.max()) >= int(n):
            raise RuntimeError(f"controlled split index out of range for n={n}")
        print(
            f"[CONTROLLED_SPLIT] {split.get('split_name')} mode={split.get('split_mode')} "
            f"core={len(core)} calib={len(calib)} val={len(val)} source={SPLIT_JSON}"
        )
        return core, calib, val

    mod = load_training_module()
    mod.select_split_indices = controlled_select_split_indices
    mod.main()


if __name__ == "__main__":
    main()
