"""Nature / SCI-style figure helpers.

Centralizes the publication-figure rules the project's `generate_*_sci_figures.py`
scripts share. Built around the nature-figure skill's contract:

- Editable SVG text (svg.fonttype = 'none') + TrueType PDF (pdf.fonttype = 42)
- Arial / DejaVu Sans / Liberation Sans fallback chain
- Single restrained palette per figure (one neutral family + one signal family)
- Panel labels separated from descriptive titles
- SVG as the primary export, PDF and PNG/TIFF as secondary artifacts

The defaults target dense journal-width multi-panel figures
(font_size ~= 7-8 pt, axes_linewidth ~= 0.8). See the skill's
`apply_publication_style()` API for slide-sized presets.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Palettes  (verbatim from nature-figure skill api.md)
# ---------------------------------------------------------------------------

PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1":   "#F6CFCB",
    "red_2":   "#E9A6A1",
    "red_strong": "#B64342",
    "neutral_light": "#CFCECE",
    "neutral_mid":   "#767676",
    "neutral_dark":  "#4D4D4D",
    "neutral_black": "#272727",
    "gold":   "#FFD700",
    "teal":   "#42949E",
    "violet": "#9A4D8E",
    "magenta": "#EA84DD",
}

DEFAULT_COLORS = [
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["neutral_light"],
]

PALETTE_NMI_PASTEL = {
    "baseline_dark": "#484878",
    "baseline_mid":  "#7884B4",
    "baseline_soft": "#B4C0E4",
    "ours_tiny":  "#E4E4F0",
    "ours_base":  "#E4CCD8",
    "ours_large": "#F0C0CC",
    "bg_lilac": "#E0E0F0",
    "bg_aqua":  "#E0F0F0",
    "bg_peach": "#F0E0D0",
    "neutral_light": "#D8D8D8",
    "neutral_mid":   "#A8A8A8",
    "neutral_dark":  "#606060",
    "delta_up":   "#2E9E44",
    "delta_down": "#E53935",
}

DEFAULT_COLORS_NMI_PASTEL = [
    PALETTE_NMI_PASTEL["baseline_dark"],
    PALETTE_NMI_PASTEL["baseline_mid"],
    PALETTE_NMI_PASTEL["baseline_soft"],
    PALETTE_NMI_PASTEL["ours_tiny"],
    PALETTE_NMI_PASTEL["ours_base"],
    PALETTE_NMI_PASTEL["ours_large"],
]


# ---------------------------------------------------------------------------
# Project-specific semantic palettes
# ---------------------------------------------------------------------------

# Adversarial attacks. Five entries form one cohesive family:
# white-box gradient attacks span a baseline blue gradient (FGSM -> PGD -> AutoPGD),
# with a signal red reserved for the black-box outlier (SquareAttack)
# and a warm pastel for the geometric attack (DeepFool).
ATTACK_COLORS = {
    "FGSM":         PALETTE_NMI_PASTEL["baseline_soft"],
    "PGD":          PALETTE_NMI_PASTEL["baseline_mid"],
    "AutoPGD":      PALETTE_NMI_PASTEL["baseline_dark"],
    "SquareAttack": PALETTE["red_strong"],
    "Square":       PALETTE["red_strong"],   # alias used in display labels
    "DeepFool":     PALETTE["violet"],
    "HopSkipJump":  PALETTE["teal"],
    "CW":           PALETTE["gold"],
    "UAP":          PALETTE["magenta"],   # white-box, image-agnostic (universal)
}

# Defense methods. Standard model is neutral; main defenses (PGD-AT / TRADES)
# get the unified blue family; preprocessor baselines fan out across teal/violet/gold
# so they stay distinguishable but visibly separate from the main defenses.
METHOD_COLORS = {
    "Standard":         PALETTE["neutral_dark"],
    "PGD-AT":           PALETTE["blue_main"],
    "TRADES":           PALETTE["blue_secondary"],
    "MART":             PALETTE["green_3"],
    "SpatialSmoothing": PALETTE["teal"],
    "JpegCompression":  PALETTE["violet"],
    "FeatureSqueezing": PALETTE["gold"],
}

# Clean vs adversarial signal. Clean = neutral, adversarial = signal red.
SIGNAL_COLORS = {
    "Clean":     PALETTE["neutral_dark"],
    "Adv":       PALETTE["red_strong"],
    "Correct":   PALETTE_NMI_PASTEL["delta_up"],
    "Incorrect": PALETTE_NMI_PASTEL["delta_down"],
}

# Per-class colors. Specific medical labels keep readable, neutral hues
# (green/red are reserved for directional cues only). Falls back to
# DEFAULT_COLORS for unknown class names (see :func:`class_color`).
CLASS_COLORS = {
    # chest_xray_pneumonia
    "Normal":    PALETTE_NMI_PASTEL["baseline_soft"],
    "Pneumonia": PALETTE["blue_main"],
    "NORMAL":    PALETTE_NMI_PASTEL["baseline_soft"],
    "PNEUMONIA": PALETTE["blue_main"],
    # oct2017
    "CNV":       PALETTE["blue_main"],
    "DME":       PALETTE["teal"],
    "DRUSEN":    PALETTE["violet"],
    # isic2020
    "benign":    PALETTE_NMI_PASTEL["baseline_soft"],
    "malignant": PALETTE["red_strong"],
}


def class_color(name: str, fallback_index: int = 0) -> str:
    """Return a deterministic color for a class name, falling back to DEFAULT_COLORS."""
    if name in CLASS_COLORS:
        return CLASS_COLORS[name]
    return DEFAULT_COLORS[fallback_index % len(DEFAULT_COLORS)]


# ---------------------------------------------------------------------------
# rcParams contract
# ---------------------------------------------------------------------------

def apply_publication_style(
    font_size: float = 7.5,
    axes_linewidth: float = 0.8,
    grid: bool = False,
    use_tex: bool = False,
) -> None:
    """Apply Nature-style rcParams. Call once before creating any figure.

    Parameters
    ----------
    font_size : float
        Base font size in points. Defaults to 7.5 (dense journal-width multi-panels).
        Use 8-10 for compact figures, 14-16 for slide / poster panels.
    axes_linewidth : float
        Width of axis spines in points. 0.8 matches Nature single-column figures.
    grid : bool
        If True, show a faint horizontal grid. Off by default; the skill recommends
        clean spines without grid for Nature-family figures.
    use_tex : bool
        Enable LaTeX rendering. Only set True when LaTeX is actually installed
        and labels need math typography.
    """
    # Mandatory: editable text in SVG.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"   # keeps text as <text> nodes, not paths
    plt.rcParams["pdf.fonttype"] = 42       # editable TrueType in PDF
    plt.rcParams["ps.fonttype"] = 42

    # Layout and style.
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "axes.linewidth": axes_linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": font_size - 0.5,
        "ytick.labelsize": font_size - 0.5,
        "xtick.major.width": axes_linewidth,
        "ytick.major.width": axes_linewidth,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": font_size - 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.4,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.6,
    })

    if grid:
        plt.rcParams.update({
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": PALETTE_NMI_PASTEL["neutral_light"],
            "grid.linewidth": 0.4,
            "grid.alpha": 0.6,
        })
    else:
        plt.rcParams["axes.grid"] = False

    if use_tex:
        plt.rcParams["text.usetex"] = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_dark(hex_color: str, threshold: int = 128) -> bool:
    """Return True if `hex_color` is dark (use white text on it)."""
    c = hex_color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < threshold


def add_panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.18,
    y: float = 1.06,
    fontsize: float = 9,
    color: str = "black",
    fontweight: str = "bold",
) -> None:
    """Place a Nature-style panel label near the axes' top-left corner.

    Use lowercase letters ('a', 'b', 'c', ...) per Nature convention.
    For dark image plates, move the label inside the axes and switch to white::

        add_panel_label(ax, "a", x=0.02, y=0.96, color="white")
    """
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        ha="left",
        va="bottom",
    )


def style_dark_image_ax(ax: plt.Axes, facecolor: str = "black") -> plt.Axes:
    """Prepare an axes for microscopy / rendering plates."""
    ax.set_facecolor(facecolor)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


# ---------------------------------------------------------------------------
# Reusable plot primitives
# ---------------------------------------------------------------------------

def make_grouped_bar(
    ax: plt.Axes,
    categories,
    series,
    labels,
    *,
    ylabel: str | None = None,
    colors=None,
    annotate: bool = False,
    bar_width: float = 0.78,
    edgecolor: str = "black",
    edge_linewidth: float = 0.45,
    annotate_fontsize: float = 6.5,
    error_kw: dict | None = None,
):
    """Grouped bar chart with Nature-style outlines.

    Parameters
    ----------
    categories : list[str]      x-axis category names
    series     : list[arraylike] one array per group, each length len(categories)
    labels     : list[str]      legend label per group
    colors     : list[str] | None  defaults to DEFAULT_COLORS_NMI_PASTEL
    bar_width  : float          total width occupied by all bars in one category
    """
    import numpy as np

    if colors is None:
        colors = DEFAULT_COLORS_NMI_PASTEL
    n_groups = len(series)
    n_cats = len(categories)
    w = bar_width / n_groups
    x = np.arange(n_cats)

    containers = []
    for i, (vals, label, color) in enumerate(zip(series, labels, colors)):
        offset = (i - (n_groups - 1) / 2) * w
        bars = ax.bar(
            x + offset, vals,
            width=w,
            label=label,
            color=color,
            edgecolor=edgecolor,
            linewidth=edge_linewidth,
            error_kw=error_kw or {},
        )
        containers.append(bars)
        if annotate:
            for bar, val in zip(bars, vals):
                if val is None or (isinstance(val, float) and (val != val)):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=annotate_fontsize,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    if ylabel:
        ax.set_ylabel(ylabel)
    return containers


def make_trend(
    ax: plt.Axes,
    x,
    y_series,
    labels,
    *,
    colors=None,
    linestyles=None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    show_shadow: bool = False,
    shadow_alpha: float = 0.15,
    lw: float = 1.3,
    marker: str = "o",
    markersize: float = 3.5,
):
    """Multi-line trend plot. y_series can be 1D or 2D (rows = runs)."""
    import numpy as np

    if colors is None:
        colors = DEFAULT_COLORS
    if linestyles is None:
        linestyles = ["-"] * len(y_series)

    for y, label, color, ls in zip(y_series, labels, colors, linestyles):
        y = np.asarray(y, dtype=float)
        if y.ndim == 2:
            mean, std = y.mean(0), y.std(0)
        else:
            mean, std = y, None
        ax.plot(
            x, mean,
            color=color, lw=lw,
            linestyle=ls,
            marker=marker, markersize=markersize,
            markeredgecolor=color, markerfacecolor=color,
            label=label,
        )
        if show_shadow and std is not None:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=shadow_alpha)

    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def make_heatmap(
    ax: plt.Axes,
    matrix,
    *,
    x_labels=None,
    y_labels=None,
    cmap: str = "Blues",
    vmin=None,
    vmax=None,
    cbar_label: str | None = None,
    annotate: bool = False,
    fmt: str = "{:.2f}",
    annotate_fontsize: float = 6.5,
):
    """2D heatmap with optional cell annotations using luminance-aware text color."""
    import numpy as np

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    if cbar_label:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
        cbar.outline.set_linewidth(0.6)
    if x_labels:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
    if y_labels:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
    if annotate:
        cmap_obj = plt.get_cmap(cmap)
        norm = mpl.colors.Normalize(
            vmin=matrix.min() if vmin is None else vmin,
            vmax=matrix.max() if vmax is None else vmax,
        )
        for (i, j), val in np.ndenumerate(matrix):
            r, g, b, _ = cmap_obj(norm(val))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            color = "white" if lum < 0.5 else "black"
            ax.text(
                j, i, fmt.format(val),
                ha="center", va="center",
                fontsize=annotate_fontsize, color=color,
            )
    ax.set_frame_on(False)
    return im


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

DEFAULT_FORMATS = ("svg", "pdf", "png", "tiff")
"""SVG is primary, PDF is vector print, PNG/TIFF are raster preview/submission fallbacks."""


def finalize_figure(
    fig: plt.Figure,
    out_path: str | os.PathLike,
    *,
    formats=DEFAULT_FORMATS,
    dpi: int = 600,
    pad: float | None = None,
    bbox_inches: str | None = None,
    close: bool = True,
) -> list[Path]:
    """Save a figure to disk as SVG (primary), plus the requested fallback formats.

    Parameters
    ----------
    out_path : path-like
        Output stem (extension is ignored / replaced).
    formats : iterable[str]
        Output extensions. Default is ('svg', 'pdf', 'png'). 'tiff' is also accepted.
    dpi : int
        Used for raster outputs (png/tiff). 600 is suitable for journal submissions.
    pad : float | None
        If given, calls fig.tight_layout(pad=pad) before saving.
    bbox_inches : str | None
        Override savefig bbox_inches (e.g. 'tight'). Defaults to current rcParams.
    close : bool
        Close the figure after saving. Default True to free memory in batch runs.

    Returns
    -------
    list[Path]
        Saved file paths in the order they were written.
    """
    base = Path(out_path)
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    if pad is not None:
        try:
            fig.tight_layout(pad=pad)
        except Exception:
            pass

    saved = []
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        kwargs = {}
        if fmt in {"png", "tiff", "tif", "jpg", "jpeg"}:
            kwargs["dpi"] = dpi
        if bbox_inches is not None:
            kwargs["bbox_inches"] = bbox_inches
        fig.savefig(out, **kwargs)
        saved.append(out)

    if close:
        plt.close(fig)
    return saved


def save_or_show_figure(
    fig: plt.Figure,
    save_path: str | os.PathLike | None = None,
    *,
    formats=DEFAULT_FORMATS,
    dpi: int = 600,
    pad: float | None = 1.0,
) -> list[Path]:
    """Save using the publication export contract, or show interactively.

    Legacy plotting helpers accept a single ``save_path``. This adapter keeps that
    API while writing all publication formats from the same output stem.
    """
    if save_path:
        return finalize_figure(fig, save_path, formats=formats, dpi=dpi, pad=pad)
    plt.show()
    return []


__all__ = [
    "PALETTE",
    "DEFAULT_COLORS",
    "PALETTE_NMI_PASTEL",
    "DEFAULT_COLORS_NMI_PASTEL",
    "ATTACK_COLORS",
    "METHOD_COLORS",
    "SIGNAL_COLORS",
    "CLASS_COLORS",
    "class_color",
    "apply_publication_style",
    "is_dark",
    "add_panel_label",
    "style_dark_image_ax",
    "make_grouped_bar",
    "make_trend",
    "make_heatmap",
    "finalize_figure",
    "save_or_show_figure",
    "DEFAULT_FORMATS",
]
