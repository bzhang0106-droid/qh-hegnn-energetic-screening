"""
Rebuild the final 3D Pareto-surface Figure 1A / Figure 5 concept plot.

This script updates the original Plotly `plot_figure1A.py` idea to the final
frozen 5432-molecule database. The plot keeps the old manuscript-facing visual
grammar: grey baseline space, smooth Pareto surface, red sweet-spot candidates,
and black industrial benchmarks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "curated_molecule_clean_v1" / "old_dataset_molecule_clean.csv"
TOP20_CSV = ROOT / "results" / "final_global_top20" / "Table_Final_Global_Top20_Structure_Property_Synthesizability_10D.csv"
PARETO_FRONT_CSV = ROOT / "results" / "final_global_top20" / "Table_S_Figure5_Final_10D_Pareto_Front.csv"

OUT_DIR = ROOT / "results" / "final_global_top20"
PACKAGE_MAIN_FIG_DIR = ROOT / "manuscript_npJ" / "final_submission_package_AL08_20260605" / "main_figures"
PACKAGE_SI_TABLE_DIR = ROOT / "manuscript_npJ" / "final_submission_package_AL08_20260605" / "si_tables"

OUT_BASENAME = "Figure_5_Final_3D_Pareto_Surface_Current_Data"
OUT_SURFACE_SOURCE = "Table_S_Figure5_3D_Pareto_Surface_Source.csv"
OUT_TOP20_SOURCE = "Table_S_Figure5_3D_Top20_Source.csv"
STATIC_CROP_BOX = (180, 80, 3650, 2705)
STATIC_WIDTH_PX = 4000
STATIC_DPI = (600, 600)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def add_candidate_ids(top20: pd.DataFrame) -> pd.DataFrame:
    top20 = top20.copy()
    if "Final_Candidate_ID" not in top20.columns:
        rank_col = "Final_Global_Rank" if "Final_Global_Rank" in top20.columns else None
        if rank_col:
            fallback = pd.Series(np.arange(1, len(top20) + 1), index=top20.index)
            ranks = pd.to_numeric(top20[rank_col], errors="coerce").fillna(fallback)
        else:
            ranks = pd.Series(np.arange(1, len(top20) + 1), index=top20.index)
        top20.insert(1, "Final_Candidate_ID", [f"C{int(r):02d}" for r in ranks])
    return top20


def minmax(series: pd.Series, maximize: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    lo = values.min()
    hi = values.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        scaled = pd.Series(np.full(len(values), 0.5), index=series.index)
    else:
        scaled = (values - lo) / (hi - lo)
    return scaled if maximize else 1.0 - scaled


def get_pareto_front(df: pd.DataFrame, max_cols: list[str], min_cols: list[str], eps: float = 1e-12) -> pd.DataFrame:
    clean = df.dropna(subset=max_cols + min_cols).copy()
    values = np.column_stack(
        [-clean[c].astype(float).to_numpy() for c in max_cols]
        + [clean[c].astype(float).to_numpy() for c in min_cols]
    )
    n = len(clean)
    is_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_front[i]:
            continue
        diff = values - values[i]
        dominated = np.any(np.all(diff <= eps, axis=1) & np.any(diff < -eps, axis=1))
        if dominated:
            is_front[i] = False
    return clean.loc[is_front].copy()


def prepare_final_database() -> pd.DataFrame:
    require_file(DATA_CSV)
    df = pd.read_csv(DATA_CSV)
    out = df.copy()
    out["HOF"] = num(out, "Heat_of_Formation(kcal/mol)")
    out["Gap"] = num(out, "HOMO_LUMO_Gap(eV)")
    out["SA"] = num(out, "SAscore") if "SAscore" in out.columns else num(out, "SA_Score")
    out["D_final"] = num(out, "Final_Detonation_D(km/s)")
    out["density_objective"] = num(out, "Final_Detonation_Density_Used(g/cm3)")
    out["rho_BCP"] = num(out, "Trigger_Bond_Rho")
    out["BDE"] = num(out, "Vertical_BDE(kcal/mol)")
    out["VS_max_num"] = num(out, "VS_max")
    out["Sigma2_num"] = num(out, "Sigma2_tot")
    out["Nu_num"] = num(out, "Nu")
    out["MW"] = num(out, "Molecular_Weight")
    return out.dropna(subset=["HOF", "Gap", "SA"]).copy()


def prepare_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    if PARETO_FRONT_CSV.exists():
        front = pd.read_csv(PARETO_FRONT_CSV)
        return front.rename(columns={"Figure5_Screening_Score_10D": "Score"}).copy()

    max_cols = ["density_objective", "HOF", "Gap", "rho_BCP", "BDE"]
    min_cols = ["SA", "VS_max_num", "Sigma2_num", "Nu_num", "MW"]
    front = get_pareto_front(df, max_cols=max_cols, min_cols=min_cols)
    score = pd.Series(np.zeros(len(front)), index=front.index, dtype=float)
    for col in max_cols:
        score += minmax(front[col], maximize=True)
    for col in min_cols:
        score += minmax(front[col], maximize=False)
    front["Score"] = score / (len(max_cols) + len(min_cols))
    front["Figure5_Target_Zone"] = (front["D_final"] >= 8.0) & (front["rho_BCP"] >= 0.165)
    return front


def prepare_top20() -> pd.DataFrame:
    require_file(TOP20_CSV)
    top20 = add_candidate_ids(pd.read_csv(TOP20_CSV))
    top20["HOF"] = num(top20, "Heat_of_Formation(kcal/mol)")
    top20["Gap"] = num(top20, "HOMO_LUMO_Gap(eV)")
    top20["SA"] = num(top20, "SA_Score")
    top20["D_final"] = num(top20, "Final_Global_D(km/s)")
    top20["rho_BCP"] = num(top20, "Trigger_Bond_Rho")
    return top20.dropna(subset=["HOF", "Gap", "SA"]).copy()


def benchmark_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": ["TNT", "RDX", "HMX", "CL-20", "TATB"],
            "HOF": [-16.0, 14.7, 17.9, 87.0, -33.4],
            "Gap": [3.8, 5.5, 5.5, 4.0, 3.2],
            "SA": [2.5, 3.5, 3.8, 5.5, 2.2],
        }
    )


def fit_surface(surface_source: pd.DataFrame, top20: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_source = surface_source[["HOF", "Gap", "SA"]].copy()
    focal_source = fit_source.copy()
    if top20 is not None and not top20.empty:
        top_source = top20[["HOF", "Gap", "SA"]].copy()
        fit_source = pd.concat([fit_source, top_source], ignore_index=True)
        focal_source = top_source

    fit_source = fit_source.dropna(subset=["HOF", "Gap", "SA"])
    px = fit_source["HOF"].to_numpy(dtype=float)
    py = fit_source["Gap"].to_numpy(dtype=float)
    pz = fit_source["SA"].to_numpy(dtype=float)
    px_mean, px_std = px.mean(), px.std() if px.std() > 0 else 1.0
    py_mean, py_std = py.mean(), py.std() if py.std() > 0 else 1.0
    px_norm = (px - px_mean) / px_std
    py_norm = (py - py_mean) / py_std

    design = np.c_[np.ones(len(px)), px_norm, py_norm, px_norm**2, py_norm**2, px_norm * py_norm]
    coeffs, _, _, _ = np.linalg.lstsq(design, pz, rcond=None)

    focal_x = pd.to_numeric(focal_source["HOF"], errors="coerce").dropna()
    focal_y = pd.to_numeric(focal_source["Gap"], errors="coerce").dropna()
    x_center = float(np.nanmedian(focal_x) - 22.0)
    y_center = float(np.nanmedian(focal_y))
    rx = float(np.clip((focal_x.quantile(0.90) - focal_x.quantile(0.10)) / 2.0 + 55.0, 78.0, 105.0))
    ry = float(np.clip((focal_y.quantile(0.90) - focal_y.quantile(0.10)) / 2.0 + 0.70, 1.05, 1.45))

    radius = np.linspace(0.0, 1.0, 110)[:, None]
    theta = np.linspace(0.0, 2.0 * np.pi, 260)[None, :]
    x_grid = x_center + rx * radius * np.cos(theta)
    y_grid = y_center + ry * radius * np.sin(theta)
    x_norm = (x_grid - px_mean) / px_std
    y_norm = (y_grid - py_mean) / py_std
    z_grid = (
        coeffs[0]
        + coeffs[1] * x_norm
        + coeffs[2] * y_norm
        + coeffs[3] * x_norm**2
        + coeffs[4] * y_norm**2
        + coeffs[5] * x_norm * y_norm
    )

    return x_grid, y_grid, z_grid


def build_figure(df: pd.DataFrame, front: pd.DataFrame, top20: pd.DataFrame, surface_source: pd.DataFrame) -> go.Figure:
    base = df.sample(min(2600, len(df)), random_state=42)
    bench = benchmark_table()
    x_grid, y_grid, z_grid = fit_surface(surface_source, top20)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=base["HOF"],
            y=base["Gap"],
            z=base["SA"],
            mode="markers",
            name="Baseline Space",
            marker=dict(size=3.0, color="#95A5A6", opacity=0.18, symbol="circle", line=dict(width=0)),
            hovertemplate="HOF=%{x:.1f}<br>Gap=%{y:.2f}<br>SA=%{z:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            colorscale=[
                [0.0, "#7CB342"],
                [0.35, "#FDD835"],
                [0.70, "#FB8C00"],
                [1.0, "#C62828"],
            ],
            opacity=0.84,
            showscale=False,
            lighting=dict(ambient=0.62, diffuse=0.86, specular=0.22, roughness=0.58),
            name="Pareto Surface",
            hoverinfo="skip",
            contours=dict(
                x=dict(show=False),
                y=dict(show=False),
                z=dict(show=False),
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=top20["HOF"],
            y=top20["Gap"],
            z=top20["SA"],
            mode="markers",
            name="HELS Pareto Sweet Spot",
            marker=dict(size=7.0, color="#D32F2F", opacity=1.0, symbol="circle", line=dict(color="white", width=1.3)),
            text=top20["Final_Candidate_ID"],
            hovertemplate="%{text}<br>HOF=%{x:.1f}<br>Gap=%{y:.2f}<br>SA=%{z:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=bench["HOF"],
            y=bench["Gap"],
            z=bench["SA"],
            mode="markers",
            name="Benchmarks",
            marker=dict(size=6.2, color="#000000", symbol="diamond", line=dict(color="white", width=1.2)),
            hovertemplate="%{text}<br>HOF=%{x:.1f}<br>Gap=%{y:.2f}<br>SA=%{z:.2f}<extra></extra>",
            text=bench["Name"],
        )
    )
    for _, row in bench.iterrows():
        fig.add_trace(
            go.Scatter3d(
                x=[row["HOF"]],
                y=[row["Gap"]],
                z=[row["SA"]],
                mode="text",
                text=[f"  {row['Name']}"],
                textposition="middle right",
                textfont=dict(size=15, color="black", family="Arial, sans-serif"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text="Energy: H_f (kcal/mol)", font=dict(size=22)),
                backgroundcolor="white",
                gridcolor="#B8B8B8",
                gridwidth=2,
                showbackground=True,
                zerolinecolor="#A6A6A6",
                zerolinewidth=2,
                showline=True,
                linecolor="#696969",
                linewidth=2,
                range=[-75, 260],
                tickfont=dict(size=18, color="#111111"),
                tickvals=[-50, 0, 50, 100, 150],
            ),
            yaxis=dict(
                title=dict(text="Stability: E_gap (eV)", font=dict(size=22)),
                backgroundcolor="white",
                gridcolor="#B8B8B8",
                gridwidth=2,
                showbackground=True,
                zerolinecolor="#A6A6A6",
                zerolinewidth=2,
                showline=True,
                linecolor="#696969",
                linewidth=2,
                range=[2.15, 6.8],
                tickfont=dict(size=18, color="#111111"),
                tickvals=[3, 4, 5, 6],
            ),
            zaxis=dict(
                title=dict(text="Synthesizability: SA Score", font=dict(size=22)),
                backgroundcolor="white",
                gridcolor="#B8B8B8",
                gridwidth=2,
                showbackground=True,
                zerolinecolor="#A6A6A6",
                zerolinewidth=2,
                showline=True,
                linecolor="#696969",
                linewidth=2,
                range=[2.0, 6.3],
                tickfont=dict(size=18, color="#111111"),
                tickvals=[2, 3, 4, 5, 6],
            ),
            camera=dict(eye=dict(x=1.55, y=-1.35, z=0.72)),
            aspectmode="manual",
            aspectratio=dict(x=1.35, y=0.92, z=0.78),
            domain=dict(x=[0.0, 0.96], y=[0.05, 1.0]),
        ),
        font=dict(family="Arial, Helvetica, sans-serif", size=19, color="black"),
        margin=dict(l=0, r=96, b=96, t=8),
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.66,
            itemsizing="constant",
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=18),
        ),
        paper_bgcolor="white",
        width=2000,
        height=1500,
    )
    return fig


def postprocess_static_assets(out_dir: Path) -> None:
    """Trim static whitespace while keeping the full interactive HTML untouched."""
    png_path = out_dir / f"{OUT_BASENAME}.png"
    if not png_path.exists():
        return
    try:
        from PIL import Image
    except Exception as exc:
        print(f"[WARN] Static whitespace trimming skipped because Pillow is unavailable: {exc}")
        return

    image = Image.open(png_path).convert("RGB")
    if image.size == (STATIC_WIDTH_PX, 3026):
        return
    if image.size != (4000, 3000):
        print(f"[WARN] Static whitespace trimming skipped for unexpected image size: {image.size}")
        return

    cropped = image.crop(STATIC_CROP_BOX)
    height = round(STATIC_WIDTH_PX * cropped.height / cropped.width)
    final_image = cropped.resize((STATIC_WIDTH_PX, height), Image.Resampling.LANCZOS)
    final_image.save(png_path, dpi=STATIC_DPI, optimize=True)
    final_image.save(out_dir / f"{OUT_BASENAME}.tif", dpi=STATIC_DPI, compression="tiff_lzw")
    final_image.save(out_dir / f"{OUT_BASENAME}.pdf", resolution=600.0)


def write_outputs(fig: go.Figure, surface_source: pd.DataFrame, top20: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_SI_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    html_path = OUT_DIR / f"{OUT_BASENAME}.html"
    fig.write_html(html_path, include_plotlyjs=True)

    for suffix in ["png", "pdf", "svg"]:
        out = OUT_DIR / f"{OUT_BASENAME}.{suffix}"
        package_out = PACKAGE_MAIN_FIG_DIR / out.name
        try:
            fig.write_image(out, width=2000, height=1500, scale=2)
            fig.write_image(package_out, width=2000, height=1500, scale=2)
        except Exception as exc:
            print(f"[WARN] Static Plotly export skipped for {suffix}: {exc}")
    postprocess_static_assets(OUT_DIR)
    postprocess_static_assets(PACKAGE_MAIN_FIG_DIR)

    package_html = PACKAGE_MAIN_FIG_DIR / html_path.name
    fig.write_html(package_html, include_plotlyjs=True)

    surface_source.to_csv(OUT_DIR / OUT_SURFACE_SOURCE, index=False)
    surface_source.to_csv(PACKAGE_SI_TABLE_DIR / OUT_SURFACE_SOURCE, index=False)
    top20.to_csv(OUT_DIR / OUT_TOP20_SOURCE, index=False)
    top20.to_csv(PACKAGE_SI_TABLE_DIR / OUT_TOP20_SOURCE, index=False)


def main() -> None:
    print("Rendering final 3D Pareto surface from current frozen data...")
    df = prepare_final_database()
    front = prepare_pareto_front(df)
    top20 = prepare_top20()
    if "Figure5_Target_Zone" in front.columns:
        target_mask = front["Figure5_Target_Zone"].astype(bool)
    else:
        target_mask = (front["D_final"] >= 8.0) & (front["rho_BCP"] >= 0.165)
    target_front = front[target_mask].copy()
    if target_front.empty:
        target_front = front.copy()
    surface_source = target_front.sort_values("Score", ascending=False).head(80).copy()

    fig = build_figure(df, front, top20, surface_source)
    write_outputs(fig, surface_source, top20)
    print("Final 3D Pareto surface generated")
    print(f"database_rows={len(df)}")
    print(f"pareto_front_rows={len(front)}")
    print(f"surface_source_rows={len(surface_source)}")
    print(f"top20_rows={len(top20)}")
    print(f"output_basename={OUT_BASENAME}")


if __name__ == "__main__":
    main()
