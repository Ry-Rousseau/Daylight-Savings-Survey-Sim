"""Visualization theme — the single source of colors, labels, and the plotnine base
(Layer 4 reporting). Per conventions.md, figures pull colors/labels from here, never
raw hex at the call site, so every figure reads as one system.

Palette: Okabe-Ito (colorblind-safe by construction; the categorical hues were run
through the dataviz validator — the purple/green pair sits in the 6-8 CVD floor band,
so every figure carries direct labels + a legend as the required secondary encoding).
Stance hues are semantic: warm = later/evening light (DST), cool = earlier/morning
light (standard).
"""
from __future__ import annotations

import os

from plotnine import (
    element_blank,
    element_line,
    element_rect,
    element_text,
    scale_fill_manual,
    theme,
    theme_minimal,
)

# --- stance palette (fixed order; semantic, CVD-safe) --------------------------
# Canonical long labels (YouGov Q4) and the sim's short labels both map to one color.
STANCE_COLORS: dict[str, str] = {
    "Permanent DST": "#E69F00",       # orange — later/evening light
    "Permanent Standard": "#0072B2",  # blue — earlier/morning light
    "Keep switching": "#009E73",      # green — status quo
    "No preference": "#CC79A7",       # purple
    "Not sure": "#999999",            # neutral gray (a non-answer, not a hue)
    # sim short labels (same colors)
    "DST": "#E69F00",
    "STD": "#0072B2",
    "SWITCH": "#009E73",
    "NOPREF": "#CC79A7",
    # YouGov Q2 (eliminate?) labels reuse the ramp
    "Yes eliminate": "#009E73",
    "No": "#0072B2",
}

STANCE_ORDER = ["Permanent DST", "Permanent Standard", "Keep switching", "No preference", "Not sure"]

# Model display order + labels for the calibration/ablation comparisons.
MODEL_ORDER = ["YouGov", "32b", "qwenmax", "qwenmax_reason", "sonnet"]
MODEL_LABELS = {
    "YouGov": "YouGov (real)",
    "32b": "Qwen3-32B",
    "qwenmax": "Qwen3.7-Max",
    "qwenmax_reason": "Qwen3.7-Max\n+reasoning",
    "sonnet": "Claude\nSonnet-5",
}

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e6e6e3"
REFERENCE = "#1a1a1a"  # for a ground-truth reference line/marker (YouGov)


def theme_polis(base_size: int = 12):
    """Clean, presentation-legible plotnine theme: recessive grid, no chartjunk."""
    return theme_minimal(base_size=base_size) + theme(
        text=element_text(color=INK),
        plot_title=element_text(size=base_size + 4, weight="bold", ha="left"),
        plot_subtitle=element_text(size=base_size, color=MUTED, ha="left"),
        axis_title=element_text(size=base_size, color=MUTED),
        panel_grid_major=element_line(color=GRID, size=0.4),
        panel_grid_minor=element_blank(),
        panel_background=element_rect(fill="#fcfcfb", color=None),
        plot_background=element_rect(fill="white", color=None),
        strip_text=element_text(size=base_size, weight="bold", color=INK),
        legend_title=element_text(size=base_size - 1, color=MUTED),
        legend_key=element_blank(),
        figure_size=(8, 5),
        dpi=150,
    )


def fill_stance(**kwargs):
    """scale_fill for stances — colors follow the stance, never its rank."""
    return scale_fill_manual(values=STANCE_COLORS, **kwargs)


def save_fig(plot, name: str, *, width: float = 8, height: float = 5, out_dir: str = "output/figures"):
    """Save a plotnine figure to output/figures (stable name, 150 dpi)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    plot.save(path, width=width, height=height, dpi=150, verbose=False)
    return path
