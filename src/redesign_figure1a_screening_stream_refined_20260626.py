from __future__ import annotations

from pathlib import Path

import math

import cairosvg
import drawsvg as draw
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("None of these paths exists: " + "; ".join(str(p) for p in candidates))


def preferred_output_dir(root: Path) -> Path:
    if (root / "main_figures").exists():
        return root / "main_figures"
    if (root / "figures").exists():
        return root / "figures" / "main"
    return root / "main_figures"


OUT_DIR = preferred_output_dir(ROOT)
NPJ_OUT_DIR = OUT_DIR
SOURCE_DIR = first_existing_path(
    [
        ROOT / "supplementary_information" / "data",
        ROOT / "data" / "processed",
        OUT_DIR,
    ]
)
ORIGINAL_FIG1 = first_existing_path(
    [
        OUT_DIR / "Figure_1_Workflow_and_Pareto_screening.png",
        ROOT.parent / "main_figures" / "Figure_1_Workflow_and_Pareto_screening.png",
    ]
)
DB_CSV = first_existing_path(
    [
        ROOT / "supplementary_information" / "data" / "Supplementary_Data_1_Curated_5432_Molecule_Database.csv",
        ROOT / "data" / "processed" / "Supplementary_Data_1_Curated_5432_Molecule_Database.csv",
    ]
)
RDX_PNG = OUT_DIR / "assets" / "rdx_vesta_screenshot_20260626.png"

PANEL_W = 650
PANEL_H = 472.9
PNG_W = 6500
PNG_H = 4729
PANEL_COMPOSE_W = 6100
B_CROP_X = 7000
PANEL_LABEL_SIZE = 15.0

BASE = "Figure_1a_generation_screening_stream_refined_NPJstyle_20260626"
FULL_BASE = "Figure_1_Workflow_and_Pareto_screening_stream_refined_fig1a_20260626"

INK = "#202936"
MUTED = "#697685"
GRID = "#C7D2DD"
STREAM_FILL = "#EEF1F3"
STREAM_EDGE = "#BFC8D0"
BLUE = "#30363D"
TEAL = "#8A8F98"
GOLD = "#D27A5B"
RED = "#C62828"
VIOLET = "#333333"


def add_text(d: draw.Drawing, value: str, x: float, y: float, size: float, *, anchor: str = "middle", weight: int = 400, fill: str = INK, leading: float = 1.10) -> None:
    lines = value.split("\n")
    y0 = y - (len(lines) - 1) * size * leading / 2
    for i, line in enumerate(lines):
        d.append(
            draw.Text(
                line,
                size,
                x,
                y0 + i * size * leading,
                text_anchor=anchor,
                font_family="Arial, Helvetica, sans-serif",
                font_weight=weight,
                fill=fill,
            )
        )


def stream_center(x: float) -> float:
    return 222.0


def stream_half_width(x: float) -> float:
    # Smoothly wide at the proposal reservoir, then narrows without covering the workflow glyphs.
    left = 52 * math.exp(-((x - 116) / 150.0) ** 2)
    shoulder = 17 * math.exp(-((x - 285) / 235.0) ** 2)
    tail = 15 + 4 * math.exp(-((x - 500) / 95.0) ** 2)
    return max(15.0, tail + left + shoulder)


def stream_bounds(x: float) -> tuple[float, float]:
    c = stream_center(x)
    h = stream_half_width(x)
    return c - h, c + h


def draw_stream(d: draw.Drawing) -> None:
    xs = np.linspace(45, 585, 120)
    upper = [(float(x), stream_bounds(float(x))[0]) for x in xs]
    lower = [(float(x), stream_bounds(float(x))[1]) for x in xs[::-1]]

    p = draw.Path(fill=STREAM_FILL, stroke=STREAM_EDGE, stroke_width=0.78, opacity=0.82)
    p.M(*upper[0])
    for x, y in upper[1:]:
        p.L(x, y)
    for x, y in lower:
        p.L(x, y)
    left_upper = upper[0]
    left_lower = lower[-1]
    p.C(72, left_lower[1] - 18, 72, left_upper[1] + 18, left_upper[0], left_upper[1])
    p.Z()
    d.append(p)

    inner = draw.Path(fill="#FFFFFF", stroke="none", opacity=0.42)
    inner_upper = [(float(x), stream_center(float(x)) - 0.35 * stream_half_width(float(x))) for x in xs]
    inner_lower = [(float(x), stream_center(float(x)) + 0.35 * stream_half_width(float(x))) for x in xs[::-1]]
    inner.M(*inner_upper[0])
    for x, y in inner_upper[1:]:
        inner.L(x, y)
    for x, y in inner_lower:
        inner.L(x, y)
    inner_left_upper = inner_upper[0]
    inner_left_lower = inner_lower[-1]
    inner.C(62, inner_left_lower[1] - 10, 62, inner_left_upper[1] + 10, inner_left_upper[0], inner_left_upper[1])
    inner.Z()
    d.append(inner)



def draw_top_workflows(d: draw.Drawing) -> None:
    # Eight compact distribution modules: readable as modules at print scale,
    # with enough internal structure to imply distinct generation batches.
    xs = np.linspace(78, 572, 8)
    rng = np.random.default_rng(202626)
    module_y = 82.5
    for idx, x in enumerate(xs, start=1):
        add_text(d, str(idx), float(x), 56, 9.4, weight=700, fill=INK)

        # Small deterministic histogram glyph: visible from far away, unlike point noise.
        heights = np.array([6.0, 10.5, 15.5, 20.0, 15.0, 10.0, 6.5])
        heights = heights * float(rng.uniform(0.88, 1.08))
        heights += rng.normal(0, 0.9, len(heights))
        bar_w = 3.15
        gap = 1.35
        total_w = len(heights) * bar_w + (len(heights) - 1) * gap
        x0 = float(x - total_w / 2)
        base_y = module_y + 12.0

        # A thin density ridge behind the bars gives the module a manuscript-style statistical feel.
        ridge = draw.Path(fill="#DCE5EB", stroke="#8FA0AE", stroke_width=0.36, opacity=0.50)
        ridge.M(x - 18.0, base_y - 2.0)
        ridge.C(x - 11.0, base_y - 15.0, x - 5.0, base_y - 20.5, x, base_y - 21.5)
        ridge.C(x + 5.0, base_y - 20.5, x + 11.0, base_y - 15.0, x + 18.0, base_y - 2.0)
        ridge.L(x + 18.0, base_y + 2.2)
        ridge.C(x + 8.0, base_y - 1.5, x - 8.0, base_y - 1.5, x - 18.0, base_y + 2.2)
        ridge.Z()
        d.append(ridge)
        d.append(draw.Line(x - 17.5, base_y + 2.8, x + 17.5, base_y + 2.8, stroke="#7F909D", stroke_width=0.48, opacity=0.76))

        for j, h in enumerate(heights):
            bx = x0 + j * (bar_w + gap)
            by = base_y - float(h)
            fill = "#2F4554" if j != 3 else "#B76A4E"
            opacity = 0.92 if j != 3 else 0.96
            d.append(draw.Rectangle(bx, by, bar_w, float(h), rx=0.9, fill=fill, stroke="none", opacity=opacity))

        d.append(draw.Line(float(x), base_y + 5.0, float(x), 124, stroke=GRID, stroke_width=0.72, opacity=0.68))
    d.append(draw.Line(float(xs[0]), 124, float(xs[-1]), 124, stroke=GRID, stroke_width=0.72, opacity=0.70))
    d.append(draw.Path(d="M 325 124 L 325 154", stroke="#8896A5", stroke_width=1.10, fill="none", opacity=0.9))
    d.append(draw.Path(d="M 325 154 L 319 145 L 331 145 Z", fill="#8896A5", opacity=0.9))


def draw_proposal_texture(d: draw.Drawing) -> None:
    rng = np.random.default_rng(426)
    palette = ["#30363D", "#5F6872", "#868F98", "#C9785A", "#D5A24B"]

    # Dense reservoir, clipped analytically to stream bounds.
    for _ in range(780):
        x = 52 + rng.beta(1.25, 4.2) * 285
        lo, hi = stream_bounds(float(x))
        y = stream_center(float(x)) + rng.normal(0, (hi - lo) * 0.22)
        if lo + 5 <= y <= hi - 5:
            d.append(draw.Circle(float(x), float(y), float(rng.uniform(0.45, 1.25)), fill=palette[rng.integers(0, len(palette))], opacity=float(rng.uniform(0.08, 0.25))))

    # Valid CHNO contraction zone.
    for _ in range(190):
        x = rng.normal(276, 38)
        lo, hi = stream_bounds(float(x))
        y = stream_center(float(x)) + rng.normal(0, (hi - lo) * 0.18)
        if 200 <= x <= 355 and lo + 6 <= y <= hi - 6:
            d.append(draw.Circle(float(x), float(y), float(rng.uniform(0.55, 1.35)), fill=GOLD, opacity=float(rng.uniform(0.14, 0.32))))

    # Background in the Pareto region from full released table.
    x, y, _, _ = real_pareto_points()
    for px, py in zip(x, y):
        lo, hi = stream_bounds(float(px))
        py = min(max(float(py), lo + 5), hi - 5)
        d.append(draw.Circle(float(px), py, 0.95, fill="#7E8A96", opacity=0.13))


def real_pareto_points() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = [
        "Heat_of_Formation(kcal/mol)",
        "HOMO_LUMO_Gap(eV)",
        "Final_Detonation_Rank",
        "Final_Detonation_D(km/s)",
        "Vertical_BDE(kcal/mol)",
    ]
    if not DB_CSV.exists():
        rng = np.random.default_rng(11)
        x = 392 + rng.random(190) * 82
        y = 185 + rng.normal(0, 14, len(x))
        tx = 404 + rng.random(18) * 68
        ty = 184 + rng.normal(0, 12, len(tx))
        return x, y, tx, ty

    df = pd.read_csv(DB_CSV, usecols=lambda c: c in cols).dropna(subset=cols)
    sample = df.sample(min(220, len(df)), random_state=21)
    hof = pd.to_numeric(sample["Heat_of_Formation(kcal/mol)"], errors="coerce")
    gap = pd.to_numeric(sample["HOMO_LUMO_Gap(eV)"], errors="coerce")
    hof_q = (hof.quantile(0.03), hof.quantile(0.97))
    gap_q = (gap.quantile(0.03), gap.quantile(0.97))
    xr = (hof - hof_q[0]) / (hof_q[1] - hof_q[0])
    yr = (gap - gap_q[0]) / (gap_q[1] - gap_q[0])
    x = 386 + np.clip(xr.to_numpy(), 0, 1) * 95
    y = 224 - np.clip(yr.to_numpy(), 0, 1) * 62

    top = df.nsmallest(18, "Final_Detonation_Rank").dropna(subset=["Final_Detonation_D(km/s)", "Vertical_BDE(kcal/mol)"])
    dval = pd.to_numeric(top["Final_Detonation_D(km/s)"], errors="coerce")
    bde = pd.to_numeric(top["Vertical_BDE(kcal/mol)"], errors="coerce")
    dx = (dval - dval.min()) / max(float(dval.max() - dval.min()), 1e-9)
    by = (bde - bde.quantile(0.05)) / max(float(bde.quantile(0.95) - bde.quantile(0.05)), 1e-9)
    rng = np.random.default_rng(56)
    tx = 414 + np.clip(dx.to_numpy(), 0, 1) * 74 + rng.normal(0, 2.2, len(dval))
    ty = 206 - (np.clip(by.to_numpy(), 0, 1) - 0.5) * 42 + rng.normal(0, 2.2, len(dval))
    return x, y, tx, ty


def draw_pareto_candidates(d: draw.Drawing) -> None:
    _, _, tx, ty = real_pareto_points()
    for px, py in zip(tx, ty):
        lo, hi = stream_bounds(float(px))
        py = min(max(float(py), lo + 6), hi - 6)
        d.append(draw.Circle(float(px), float(py), 2.45, fill=RED, stroke="white", stroke_width=0.62, opacity=0.96))


def draw_release_table(d: draw.Drawing) -> None:
    y = stream_center(540) - 25
    for j, shift in enumerate([8, 4, 0]):
        d.append(draw.Rectangle(525 + shift, y - j * 4, 55, 44, rx=2.8, fill="white", stroke="#B6C1CD", stroke_width=0.72, opacity=0.98))
    for i in range(1, 4):
        d.append(draw.Line(525, y + i * 11, 580, y + i * 11, stroke="#D3DCE5", stroke_width=0.5))
        d.append(draw.Line(525 + i * 13.75, y, 525 + i * 13.75, y + 44, stroke="#D3DCE5", stroke_width=0.5))
    for i, label in enumerate(["ID", "D", "BDE", "SA"]):
        add_text(d, label, 532 + i * 13.75, y + 8, 4.8, fill=MUTED, weight=700)
    for row in range(2):
        for col, c in enumerate(["#CDD5DE", "#7FA8CB", "#77B6B1", "#D0AA58"]):
            d.append(draw.Circle(532 + col * 13.75, y + 20 + row * 10.7, 1.75, fill=c, opacity=0.85))


def draw_3d_nitramine(d: draw.Drawing) -> None:
    if RDX_PNG.exists():
        d.append(draw.Image(222, 366, 104, 102, path=str(RDX_PNG), embed=True))


def draw_pareto_callout(d: draw.Drawing) -> None:
    # A thin black frame marks the subset that maps to panel b.
    d.append(draw.Rectangle(394, 169, 112, 187, fill="none", stroke="#111111", stroke_width=0.9, opacity=0.95))


def draw_stage_labels(d: draw.Drawing) -> None:
    stages = [
        (102, "~400k\nproposals", BLUE),
        (278, "valid\nCHNO", TEAL),
        (424, "Pareto\nsurvivors", GOLD),
        (560, "records", VIOLET),
    ]
    for x, label, c in stages:
        label_x = x
        dot_y = 318
        d.append(draw.Circle(x, dot_y, 2.7, fill=c, opacity=0.9))
        if label == "valid\nCHNO":
            label_y = 347
        elif label == "Pareto\nsurvivors":
            label_y = 344
            label_x = 448
        else:
            label_y = 351
        line_end = max(dot_y + 8, label_y - 22)
        d.append(draw.Line(x, dot_y + 5, x, line_end, stroke=c, stroke_width=0.86, opacity=0.42))
        add_text(d, label, label_x, label_y, 13.6, weight=700, fill=INK, leading=1.02)


def build_svg() -> str:
    d = draw.Drawing(PANEL_W, PANEL_H, origin=(0, 0), display_inline=False)
    d.append(draw.Rectangle(0, 0, PANEL_W, PANEL_H, fill="white"))
    add_text(d, "a", 11, 28, PANEL_LABEL_SIZE, anchor="start", weight=700)
    draw_top_workflows(d)
    draw_stream(d)
    draw_proposal_texture(d)
    draw_3d_nitramine(d)
    draw_pareto_candidates(d)
    draw_release_table(d)
    draw_pareto_callout(d)
    draw_stage_labels(d)
    return d.as_svg()


def compose_full(panel_png: Path) -> Path:
    original = Image.open(ORIGINAL_FIG1).convert("RGBA")
    b_crop = original.crop((B_CROP_X, 0, original.width, original.height))
    panel = Image.open(panel_png).convert("RGBA")
    if panel.size != (PNG_W, PNG_H):
        panel = panel.resize((PNG_W, PNG_H), Image.Resampling.LANCZOS)
    panel = panel.crop((0, 0, PANEL_COMPOSE_W, PNG_H))

    canvas = Image.new("RGBA", (PANEL_COMPOSE_W + b_crop.width, original.height), (255, 255, 255, 255))
    canvas.alpha_composite(panel, (0, 0))
    canvas.alpha_composite(b_crop, (PANEL_COMPOSE_W, 0))

    # Draw final cross-panel annotation after composition so the pointer lands on panel b.
    pointer = ImageDraw.Draw(canvas)
    b_left = PANEL_COMPOSE_W
    b_right = canvas.width - 1
    pointer.rectangle([b_left + 12, 12, b_left + 285, 235], fill=(255, 255, 255, 255))
    b_box = [b_left + 820, 235, min(b_right - 22, b_left + 5900), 4445]
    pointer.rectangle(b_box, outline=(17, 17, 17, 242), width=9)

    start = (int(506 / PANEL_W * PNG_W), int(356 / PANEL_H * PNG_H))
    end = (b_box[0], start[1])
    pointer.line([start, end], fill=(17, 17, 17, 242), width=9)
    pointer.polygon(
        [
            (end[0], end[1]),
            (end[0] - 78, end[1] - 35),
            (end[0] - 78, end[1] + 35),
        ],
        fill=(17, 17, 17, 242),
    )
    try:
        panel_label_font = ImageFont.truetype("arialbd.ttf", int(PANEL_LABEL_SIZE * PNG_W / PANEL_W))
    except OSError:
        panel_label_font = ImageFont.load_default()
    pointer.text((b_left + 78, 52), "b", fill=(17, 17, 17, 255), font=panel_label_font)

    out = OUT_DIR / f"{FULL_BASE}.png"
    canvas.convert("RGB").save(out, dpi=(600, 600))
    NPJ_OUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(NPJ_OUT_DIR / out.name, dpi=(600, 600))
    return out


def write_source() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"stage": "workflow sources", "display": "8 source point groups", "role": "retained without headline text"},
            {"stage": "raw proposal pool", "display": "~400k proposals", "role": "large dense entrance"},
            {"stage": "valid CHNO", "display": "valid CHNO", "role": "chemical-scope contraction"},
            {"stage": "Pareto survivors", "display": "Pareto survivors", "role": "real released-table background plus top-ranked points clipped to stream"},
            {"stage": "standardized records", "display": "records", "role": "release table endpoint"},
        ]
    ).to_csv(SOURCE_DIR / f"{BASE}_source.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NPJ_OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    svg_path = OUT_DIR / f"{BASE}.svg"
    png_path = OUT_DIR / f"{BASE}.png"
    pdf_path = OUT_DIR / f"{BASE}.pdf"
    svg_path.write_text(svg, encoding="utf-8")
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(png_path), output_width=PNG_W, output_height=PNG_H)
    cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(pdf_path), output_width=PNG_W, output_height=PNG_H)
    for p in [svg_path, png_path, pdf_path]:
        (NPJ_OUT_DIR / p.name).write_bytes(p.read_bytes())
    write_source()
    full = compose_full(png_path)
    print(f"svg={svg_path}")
    print(f"png={png_path}")
    print(f"pdf={pdf_path}")
    print(f"full_preview={full}")


if __name__ == "__main__":
    main()
