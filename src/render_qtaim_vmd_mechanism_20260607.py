from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FORMAL_FIG = ROOT / "Manuscript" / "02_main_figures" / "formal_v4_nature_revision"
SI_TABLES = ROOT / "Manuscript" / "04_supplementary_tables" / "01_si_tables"

ASCII_WORK = Path(os.environ.get("QH_HEGNN_VMD_WORK", str(ROOT / "work" / "vmd_render"))).resolve()
INPUTS = ASCII_WORK / "inputs"
RENDERS = ASCII_WORK / "renders"
RENDERS.mkdir(parents=True, exist_ok=True)

VMD_EXE = Path(r"D:\downloads\VMD2\vmd.exe")
BOHR_TO_ANG = 0.529177210903
WORD_SMALL_FIVE_PT = 9
PANEL_LABEL_PT = 11

PALETTE = {
    "ink": "#263238",
    "line": "#D7DEE7",
    "blue": "#6FA8DC",
    "deepblue": "#3D6FB6",
    "purple": "#7B6BC7",
    "red": "#D88A7D",
    "deepred": "#B85E59",
    "cream": "#FBFAF6",
    "grey": "#8A9099",
}

MOLECULES = [
    {
        "id": "D1",
        "label": "D-only #1",
        "molecule": "AL04_Target_0027",
        "target_rho": 0.205094,
        "D": 10.1457,
        "BDE": 32.588099,
        "role": "performance-first counterexample",
    },
    {
        "id": "Q18",
        "label": "Q18",
        "molecule": "AL05_Target_0498",
        "target_rho": 0.212925,
        "D": 9.9024,
        "BDE": 79.424061,
        "role": "QTAIM survivor",
    },
    {
        "id": "Q20",
        "label": "Q20",
        "molecule": "AL08_Target_0205",
        "target_rho": 0.213134,
        "D": 8.7916,
        "BDE": 57.695570,
        "role": "QTAIM survivor",
    },
]


def parse_critic2(path: Path, target_rho: float) -> dict:
    lines = path.read_text(errors="ignore").splitlines()
    atoms: dict[int, dict] = {}
    cps: dict[int, dict] = {}
    bonds: dict[int, tuple[int, int]] = {}
    for line in lines:
        atom = re.match(
            r"\s*(\d+)\s+([-0-9.Ee+]+)\s+([-0-9.Ee+]+)\s+([-0-9.Ee+]+)\s+\d+\s+([A-Z][a-z]?)_",
            line,
        )
        if atom:
            atoms[int(atom.group(1))] = {
                "element": atom.group(5),
                "xyz": np.array([float(atom.group(2)), float(atom.group(3)), float(atom.group(4))], dtype=float),
            }
        cp = re.match(
            r"\s*(\d+)\s+\(3,-1\)\s+bond\s+([-0-9.Ee+]+)\s+([-0-9.Ee+]+)\s+([-0-9.Ee+]+)\s+\S+\s+([-0-9.Ee+]+)\s+[-0-9.Ee+]+\s+([-0-9.Ee+]+)",
            line,
        )
        if cp:
            cps[int(cp.group(1))] = {
                "xyz": np.array([float(cp.group(2)), float(cp.group(3)), float(cp.group(4))], dtype=float),
                "rho": float(cp.group(5)),
                "laplacian": float(cp.group(6)),
            }
        bond = re.match(
            r"\s*(\d+)\s+[A-Z][a-z]?_\s+\((\d+)\)\s+[A-Z][a-z]?_\s+\((\d+)\)",
            line,
        )
        if bond:
            bonds[int(bond.group(1))] = (int(bond.group(2)), int(bond.group(3)))

    candidates = [(abs(item["rho"] - target_rho), n, item) for n, item in cps.items() if n in bonds]
    if not candidates:
        raise RuntimeError(f"No matching BCP found in {path}")
    diff, ncp, bcp = sorted(candidates, key=lambda x: x[0])[0]
    a1, a2 = bonds[ncp]
    bcp = dict(bcp)
    bcp.update({"ncp": ncp, "rho_abs_difference": diff, "atom1": a1, "atom2": a2})
    return {"atoms": atoms, "bcp": bcp}


def parse_cube(path: Path) -> dict:
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        header = [next(fh).strip() for _ in range(6)]
        nat_line = header[2].split()
        natoms = abs(int(nat_line[0]))
        origin = np.array([float(x) for x in nat_line[1:4]], dtype=float)
        axes = []
        counts = []
        for line in header[3:6]:
            parts = line.split()
            counts.append(int(parts[0]))
            axes.append([float(parts[1]), float(parts[2]), float(parts[3])])
        atom_lines = [next(fh).strip() for _ in range(natoms)]
        values = np.fromstring(fh.read(), sep=" ", dtype=float)
    counts_arr = np.array(counts, dtype=int)
    expected = int(np.prod(counts_arr))
    if values.size != expected:
        raise RuntimeError(f"{path.name}: expected {expected} cube values, got {values.size}")
    atoms = []
    for line in atom_lines:
        parts = line.split()
        atoms.append(
            {
                "Z": int(float(parts[0])),
                "charge": float(parts[1]),
                "xyz_bohr": np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=float),
            }
        )
    return {
        "origin": origin,
        "axes": np.array(axes, dtype=float),
        "counts": counts_arr,
        "atoms": atoms,
        "data": values.reshape(tuple(counts_arr)),
    }


def trilinear(cube: dict, points_bohr: np.ndarray) -> np.ndarray:
    inv_axes = np.linalg.inv(cube["axes"].T)
    frac = (points_bohr - cube["origin"]) @ inv_axes.T
    counts = cube["counts"]
    data = cube["data"]
    valid = np.all((frac >= 0) & (frac < counts - 1), axis=1)
    out = np.full(points_bohr.shape[0], np.nan, dtype=float)
    f = frac[valid]
    i0 = np.floor(f).astype(int)
    t = f - i0
    i, j, k = i0[:, 0], i0[:, 1], i0[:, 2]
    tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
    c000 = data[i, j, k]
    c100 = data[i + 1, j, k]
    c010 = data[i, j + 1, k]
    c110 = data[i + 1, j + 1, k]
    c001 = data[i, j, k + 1]
    c101 = data[i + 1, j, k + 1]
    c011 = data[i, j + 1, k + 1]
    c111 = data[i + 1, j + 1, k + 1]
    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    out[valid] = c0 * (1 - tz) + c1 * tz
    return out


def smooth2d(arr: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x * x) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    work = np.array(arr, dtype=float, copy=True)
    mask = np.isfinite(work)
    fill = np.nanmedian(work[mask]) if mask.any() else 0.0
    work[~mask] = fill
    pad_x = np.pad(work, ((0, 0), (radius, radius)), mode="edge")
    sm_x = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, pad_x)
    pad_y = np.pad(sm_x, ((radius, radius), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, pad_y)


def write_vmd_script(meta: dict, bcp: dict, atoms: dict[int, dict]) -> Path:
    qid = meta["id"]
    atom1 = atoms[bcp["atom1"]]["xyz"]
    atom2 = atoms[bcp["atom2"]]["xyz"]
    script = RENDERS / f"{qid}_render.tcl"
    script.write_text(
        f"""
set inpath \"{INPUTS.as_posix()}\"
set outpath \"{RENDERS.as_posix()}\"
color Display Background white
display projection Orthographic
display resize 1400 1050
axes location Off
color scale method BWR
material change opacity Transparent 0.38
mol new \"$inpath/{qid}_rho.cub\" type cube waitfor all
set mid [molinfo top]
mol addfile \"$inpath/{qid}_esp.cub\" type cube molid $mid waitfor all
mol delrep 0 $mid
mol representation CPK 0.40 0.14 24 18
mol color Element
mol material AOShiny
mol addrep $mid
mol representation Isosurface 0.001 0 0 0 1 1
mol color Volume 1
mol material Transparent
mol addrep $mid
mol scaleminmax $mid 1 -0.08 0.08
graphics $mid color yellow
graphics $mid sphere {{{bcp['xyz'][0]:.6f} {bcp['xyz'][1]:.6f} {bcp['xyz'][2]:.6f}}} radius 0.13 resolution 30
graphics $mid color purple
graphics $mid cylinder {{{atom1[0]:.6f} {atom1[1]:.6f} {atom1[2]:.6f}}} {{{atom2[0]:.6f} {atom2[1]:.6f} {atom2[2]:.6f}}} radius 0.055 resolution 30 filled yes
rotate x by -72
rotate z by -28
scale by 1.18
render TachyonInternal \"$outpath/{qid}_vmd.tga\"
quit
""".strip()
        + "\n",
        encoding="ascii",
    )
    return script


def run_vmd(script: Path, qid: str) -> Path:
    if not VMD_EXE.exists():
        raise FileNotFoundError(VMD_EXE)
    log = RENDERS / f"{qid}_vmd.log"
    proc = subprocess.run(
        [str(VMD_EXE), "-dispdev", "text", "-e", str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    log.write_text(proc.stdout, encoding="utf-8", errors="ignore")
    tga = RENDERS / f"{qid}_vmd.tga"
    png = RENDERS / f"{qid}_vmd.png"
    if not tga.exists():
        raise RuntimeError(f"VMD did not render {qid}; see {log}")
    Image.open(tga).convert("RGB").save(png)
    return png


def cropped_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    mask = np.any(arr < 248, axis=2)
    if not mask.any():
        return arr
    y, x = np.where(mask)
    pad = 24
    y0, y1 = max(0, y.min() - pad), min(arr.shape[0], y.max() + pad)
    x0, x1 = max(0, x.min() - pad), min(arr.shape[1], x.max() + pad)
    return arr[y0:y1, x0:x1]


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.965,
        0.955,
        label.lower(),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=PANEL_LABEL_PT,
        fontweight="bold",
        color=PALETTE["ink"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.4),
        zorder=10,
    )


def draw_slice(ax, meta: dict, critic: dict, rho_cube: dict, esp_cube: dict) -> dict:
    atoms = critic["atoms"]
    bcp = critic["bcp"]
    a1 = atoms[bcp["atom1"]]["xyz"]
    a2 = atoms[bcp["atom2"]]["xyz"]
    bcp_ang = bcp["xyz"]
    e1 = a2 - a1
    e1 = e1 / np.linalg.norm(e1)
    atom_coords = np.vstack([atom["xyz"] for atom in atoms.values()])
    centered = atom_coords - atom_coords.mean(axis=0)
    projected = centered - np.outer(centered @ e1, e1)
    _, _, vh = np.linalg.svd(projected, full_matrices=False)
    e2 = vh[0]
    if abs(np.dot(e1, e2)) > 0.8:
        e2 = np.array([0.0, 0.0, 1.0])
        e2 = e2 - np.dot(e2, e1) * e1
        e2 = e2 / np.linalg.norm(e2)

    extent_x = 2.15
    extent_y = 1.65
    nx, ny = 360, 280
    xs = np.linspace(-extent_x, extent_x, nx)
    ys = np.linspace(-extent_y, extent_y, ny)
    X, Y = np.meshgrid(xs, ys)
    points_ang = bcp_ang + X[..., None] * e1 + Y[..., None] * e2
    points_bohr = points_ang.reshape(-1, 3) / BOHR_TO_ANG
    rho = trilinear(rho_cube, points_bohr).reshape(ny, nx)
    esp = trilinear(esp_cube, points_bohr).reshape(ny, nx)
    rho = smooth2d(rho, sigma=0.75)
    esp = smooth2d(esp, sigma=0.85)
    esp_abs = np.nanpercentile(np.abs(esp), 98)
    esp_abs = max(float(esp_abs), 0.035)
    esp_cmap = mcolors.LinearSegmentedColormap.from_list(
        "soft_esp",
        ["#5A78B7", "#D9E5F4", "#FBFAF6", "#F0D6CE", "#C66B63"],
    )
    ax.contourf(
        X,
        Y,
        esp,
        levels=np.linspace(-esp_abs, esp_abs, 19),
        cmap=esp_cmap,
        norm=mcolors.TwoSlopeNorm(vmin=-esp_abs, vcenter=0.0, vmax=esp_abs),
        alpha=0.96,
        extend="both",
    )
    rho_levels = np.array([0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256])
    rho_levels = rho_levels[(rho_levels > np.nanmin(rho)) & (rho_levels < np.nanmax(rho))]
    rho_contours = ax.contour(X, Y, rho, levels=rho_levels, colors="#263238", linewidths=0.56, alpha=0.82)
    if len(rho_contours.levels):
        ax.clabel(
            rho_contours,
            rho_contours.levels[::3],
            inline=True,
            fontsize=WORD_SMALL_FIVE_PT,
            fmt=lambda v: f"{v:.3f}",
            colors="#263238",
        )
    ax.contour(X, Y, rho, levels=[0.001], colors="#5D6570", linewidths=1.05, alpha=0.95)
    neg_levels = [-0.075, -0.050, -0.025]
    pos_levels = [0.025, 0.050, 0.075]
    neg_levels = [v for v in neg_levels if np.nanmin(esp) < v < np.nanmax(esp)]
    pos_levels = [v for v in pos_levels if np.nanmin(esp) < v < np.nanmax(esp)]
    if neg_levels:
        ax.contour(X, Y, esp, levels=neg_levels, colors=PALETTE["deepblue"], linewidths=0.80, alpha=0.88, linestyles="--")
    if pos_levels:
        ax.contour(X, Y, esp, levels=pos_levels, colors=PALETTE["deepred"], linewidths=0.80, alpha=0.88)
    ax.plot([-np.dot(bcp_ang - a1, e1), np.dot(a2 - bcp_ang, e1)], [0, 0], color=PALETTE["purple"], lw=2.25, zorder=4)
    ax.scatter([0], [0], s=58, facecolor="#F0E91B", edgecolor=PALETTE["ink"], lw=0.7, zorder=7)
    ax.scatter(
        [-np.dot(bcp_ang - a1, e1), np.dot(a2 - bcp_ang, e1)],
        [0, 0],
        s=62,
        c=[PALETTE["grey"], PALETTE["blue"]],
        edgecolor="white",
        lw=0.6,
        zorder=6,
    )
    ax.text(
        0.03,
        0.10,
        f"{meta['id']}  D={meta['D']:.2f} km s$^{{-1}}$  BDE={meta['BDE']:.1f} kcal mol$^{{-1}}$",
        transform=ax.transAxes,
        fontsize=WORD_SMALL_FIVE_PT,
        color=PALETTE["ink"],
        bbox=dict(facecolor="white", edgecolor=PALETTE["line"], boxstyle="round,pad=0.22", alpha=0.82),
        zorder=8,
    )
    ax.text(
        0.03,
        0.90,
        f"$\\rho_{{BCP}}$={bcp['rho']:.3f}; $\\nabla^2\\rho$={bcp['laplacian']:.3f}",
        transform=ax.transAxes,
        fontsize=WORD_SMALL_FIVE_PT,
        color=PALETTE["ink"],
        bbox=dict(facecolor="white", edgecolor=PALETTE["line"], boxstyle="round,pad=0.20", alpha=0.78),
        zorder=8,
    )
    ax.set_xlim(-extent_x, extent_x)
    ax.set_ylim(-extent_y, extent_y)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["line"])
    return {
        "Display_ID": meta["id"],
        "Molecule": meta["molecule"],
        "Matched_Critic2_CP": bcp["ncp"],
        "Trigger_Bond": f"{atoms[bcp['atom1']]['element']}{bcp['atom1']}-{atoms[bcp['atom2']]['element']}{bcp['atom2']}",
        "Matched_CP_Rho": bcp["rho"],
        "Matched_CP_Laplacian": bcp["laplacian"],
        "Rho_abs_difference": bcp["rho_abs_difference"],
        "D_km_s": meta["D"],
        "BDE_kcal_mol": meta["BDE"],
        "Rendering_inputs": "Multiwfn rho.cub and total ESP cube; Critic2 BCP; VMD TachyonInternal",
    }


def main() -> None:
    render_rows = []
    prepared = []
    for meta in MOLECULES:
        qid = meta["id"]
        critic = parse_critic2(INPUTS / f"{qid}_critic2_cpreport.out", meta["target_rho"])
        script = write_vmd_script(meta, critic["bcp"], critic["atoms"])
        png = run_vmd(script, qid)
        rho_cube = parse_cube(INPUTS / f"{qid}_rho.cub")
        esp_cube = parse_cube(INPUTS / f"{qid}_esp.cub")
        prepared.append((meta, critic, rho_cube, esp_cube, png))

    fig = plt.figure(figsize=(12.8, 7.05))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.16, 1.0], hspace=0.18, wspace=0.10)
    for col, (meta, critic, rho_cube, esp_cube, png) in enumerate(prepared):
        ax = fig.add_subplot(gs[0, col])
        image = cropped_image(png)
        ax.imshow(image)
        ax.set_axis_off()
        panel = chr(ord("a") + col)
        add_panel_label(ax, panel)
        bcp = critic["bcp"]

        ax2 = fig.add_subplot(gs[1, col])
        row = draw_slice(ax2, meta, critic, rho_cube, esp_cube)
        add_panel_label(ax2, chr(ord("d") + col))
        row["VMD_render_png"] = str(png)
        render_rows.append(row)

    out = FORMAL_FIG / "Figure_6_QTAIM_VMD_Topology_Mechanism_NPJStyle_20260607.png"
    out_pdf = out.with_suffix(".pdf")
    fig.savefig(out, dpi=720, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    table = pd.DataFrame(render_rows)
    table.to_csv(SI_TABLES / "Table_S_QTAIM_VMD_Topology_Render_Source_20260607.csv", index=False, encoding="utf-8-sig")
    print(out)
    print(out_pdf)


if __name__ == "__main__":
    main()
