#!/usr/bin/env python3
"""
Publication-quality efficiency frontier (PCA suite): Training energy vs Macro-F1@tau*.

- Adds tau* explicitly in axis labels/title for defensibility.
- Fixes baseline key bug (macro_f1_05 vs macro_f1).
- Uses Macro-F1@tau* values (consistent with Table I).
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def pareto_mask_min_x_max_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return mask for Pareto-efficient points under: minimize x, maximize y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (x[j] <= x[i] and y[j] >= y[i]) and (x[j] < x[i] or y[j] > y[i]):
                mask[i] = False
                break
    return mask


def ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def auto_offset(x: float, y: float, xmid: float, ymid: float) -> tuple[int, int, str, str]:
    """Choose offset and alignment based on quadrant relative to midpoints."""
    dx, ha = 8, "left"
    dy, va = (7, "bottom") if y >= ymid else (-10, "top")
    return dx, dy, ha, va


def main(outdir: str = "results") -> None:
    outdir = ensure_outdir(outdir)

    # -----------------------------
    # PCA-suite results (Macro-F1@tau*)
    # -----------------------------
    baselines = [
        {"name": "LSTM", "energy": 0.1921, "macro_f1": 0.5300},
        {"name": "GRU",  "energy": 0.2001, "macro_f1": 0.5204},
    ]

    stgnn = [
        {"k": 0,  "energy": 0.4404, "macro_f1": 0.5288, "variant": "identity"},
        {"k": 1,  "energy": 0.3892, "macro_f1": 0.5232, "variant": "rel"},
        {"k": 2,  "energy": 0.3944, "macro_f1": 0.5262, "variant": "rel"},
        {"k": 3,  "energy": 0.4109, "macro_f1": 0.5348, "variant": "rel"},
        {"k": 5,  "energy": 0.4629, "macro_f1": 0.5349, "variant": "rel"},
        {"k": 8,  "energy": 0.5071, "macro_f1": 0.5354, "variant": "rel"},
        {"k": 50, "energy": 0.7308, "macro_f1": 0.5307, "variant": "rel"},
    ]

    # Pareto on STGNN only: minimize energy, maximize macro-f1
    xs = np.array([p["energy"] for p in stgnn], dtype=float)
    ys = np.array([p["macro_f1"] for p in stgnn], dtype=float)
    pareto_mask = pareto_mask_min_x_max_y(xs, ys)

    pareto_pts = [stgnn[i] for i in range(len(stgnn)) if pareto_mask[i]]
    pareto_pts = sorted(pareto_pts, key=lambda d: d["energy"])

    # -----------------------------
    # Typography (paper appropriate)
    # -----------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)

    # Palette
    c_base = "0.10"
    c_rel = "0.40"
    c_id = "0.55"
    c_pareto = "#B00020"

    # Baselines
    for b in baselines:
        ax.scatter(b["energy"], b["macro_f1"], marker="s", s=52, color=c_base, zorder=4)
        ax.annotate(b["name"], (b["energy"], b["macro_f1"]),
                    textcoords="offset points", xytext=(7, 6), ha="left", va="bottom")

    # STGNN points
    for p in stgnn:
        if p["variant"] == "identity":
            ax.scatter(p["energy"], p["macro_f1"], marker="D", s=55, color=c_id, zorder=3)
        else:
            ax.scatter(p["energy"], p["macro_f1"], marker="o", s=45, color=c_rel, zorder=3)

    # Pareto front (STGNN)
    pareto_x = [p["energy"] for p in pareto_pts]
    pareto_y = [p["macro_f1"] for p in pareto_pts]
    ax.plot(pareto_x, pareto_y, linewidth=1.8, color=c_pareto, zorder=2)

    # Ring Pareto points
    for p in pareto_pts:
        ax.scatter(p["energy"], p["macro_f1"], marker="o", s=120,
                   facecolors="none", edgecolors=c_pareto, linewidths=1.6, zorder=5)

    # Annotations
    label_ks = {0, 1, 2, 3, 5, 8, 50}
    all_x = [b["energy"] for b in baselines] + [p["energy"] for p in stgnn]
    all_y = [b["macro_f1"] for b in baselines] + [p["macro_f1"] for p in stgnn]
    xmid = (min(all_x) + max(all_x)) / 2.0
    ymid = (min(all_y) + max(all_y)) / 2.0

    manual = {
        1:  (-10, -14, "left", "top"),
        2:  (-40, -5, "left", "bottom"),
        3:  (0, 10, "right", "bottom"),
        5:  (-15, 10, "left", "bottom"),
        8:  (10, 10, "left", "bottom"),
        50: (-15, 10, "left", "bottom"),
        0:  (-10, -12, "left", "top"),
    }

    for p in stgnn:
        if p["k"] not in label_ks:
            continue
        if p["k"] in manual:
            dx, dy, ha, va = manual[p["k"]]
        else:
            dx, dy, ha, va = auto_offset(p["energy"], p["macro_f1"], xmid, ymid)

        ax.annotate(rf"$k={p['k']}$", (p["energy"], p["macro_f1"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, va=va, color="0.10")

    # Axes / labels (with tau*)
    ax.set_title(r"Efficiency frontier (PCA suite): Energy vs Macro-F1@$\,\tau^\star$")
    ax.set_xlabel("Training energy (Wh)")
    ax.set_ylabel(r"Macro-F1@$\,\tau^\star$ (held-out)")

    ax.grid(True, linewidth=0.6, alpha=0.22)
    ax.set_axisbelow(True)
    ax.set_xlim(min(all_x) - 0.03, max(all_x) + 0.04)
    ax.set_ylim(min(all_y) - 0.010, max(all_y) + 0.010)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # Legend
    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="None", markersize=7,
               markerfacecolor=c_base, markeredgecolor=c_base, label="Baselines (LSTM/GRU)"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=7,
               markerfacecolor=c_rel, markeredgecolor=c_rel, label=r"STGNN Relational Top-$k$"),
        Line2D([0], [0], marker="D", linestyle="None", markersize=7,
               markerfacecolor=c_id, markeredgecolor=c_id, label=r"STGNN Identity ($k{=}0$)"),
        Line2D([0], [0], linestyle="-", linewidth=1.8, color=c_pareto, label="Pareto front (STGNN)"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=8,
               markerfacecolor="none", markeredgecolor=c_pareto, markeredgewidth=1.6,
               label="Pareto-efficient STGNN"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.95,
              borderpad=0.6, handlelength=2.2, handletextpad=0.7)

    # Save
    pdf_path = os.path.join(outdir, "pareto_energy_macroF1_tau.pdf")
    png_path = os.path.join(outdir, "pareto_energy_macroF1_tau.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    pareto_ks = [p["k"] for p in pareto_pts]
    print(f"Pareto-efficient STGNN k values: {pareto_ks}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
