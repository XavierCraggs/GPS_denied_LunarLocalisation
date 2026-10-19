"""Report-ready figures from the analysis tables (run AFTER analyse_results.py and run_diagnostics.py summary).

Usage:  python make_report_figures.py      -> analysis/figures/*.png (300 dpi, consistent method colours)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
A = HERE / "analysis"
OUT = A / "figures"
POS_OK = 0.5

# One fixed colour per method in every figure (validated categorical palette, CVD-safe adjacent pairs).
# PnP = solid lines, silhouette = dashed -> identity never relies on colour alone.
METHODS = ["ORB", "AKAZE", "LightGlue", "SIL-edge", "SIL-oracle"]
COL = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
LS = {m: "--" if m.startswith("SIL") else "-" for m in METHODS}
MK = dict(zip(METHODS, ["o", "s", "^", "D", "v"]))
INK, GRID = "#3a3a38", "#e4e3df"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("  ", name)


def fig_cdf(df):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, scene in zip(axes, ("v1", "v2")):
        s = df[df["scene"] == scene]
        for m in METHODS:
            d = s[s["method"] == m]
            e = np.sort(d.loc[d["pose"], "pos_err_m"].to_numpy())
            ax.step(e, np.arange(1, len(e) + 1) / len(d) * 100, where="post", color=COL[m], ls=LS[m], lw=1.6, label=m)
        ax.axvline(POS_OK, color=INK, lw=0.7, ls=":")
        ax.text(POS_OK * 1.1, 3, "0.5 m", color=INK, fontsize=8)
        ax.set(xscale="log", xlim=(1e-3, 60), ylim=(0, 100), xlabel="Position error (m, log scale)", title=f"Scene {scene}")
    axes[0].set_ylabel("Frames with error ≤ x (%)")
    axes[1].legend(loc="upper left", fontsize=8)
    save(fig, "fig_cdf_position.png")


def fig_error_vs_range(df):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, scene in zip(axes, ("v1", "v2")):
        s = df[(df["scene"] == scene) & df["ok"]]
        for m in METHODS:
            d = s[s["method"] == m]
            ax.scatter(d["gt_range"], d["pos_err_m"] * 100, s=12, marker=MK[m], color=COL[m], alpha=0.8,
                       edgecolors="white", linewidths=0.4, label=m)
        ax.set(yscale="log", xlabel="Range to lander (m)", title=f"Scene {scene} (successful frames)")
    axes[0].set_ylabel("Position error (cm, log scale)")
    axes[1].legend(fontsize=8, loc="lower right")
    save(fig, "fig_error_vs_range.png")


def fig_failure_modes(df):
    cats = ["near miss (≤2 m)", "flip ~180°", "alias ~90°", "gross error", "no pose"]
    ccol = ["#86b6ef", "#d55181", "#c98500", "#6b6a66", "#c3c2b7"]
    t = df.groupby(["method", "scene", "failure"]).size().unstack("failure", fill_value=0)
    rows = [(m, s) for m in METHODS for s in ("v1", "v2")]
    t = t.reindex(rows).fillna(0)
    x = np.arange(len(rows)) + np.repeat(np.arange(len(METHODS)) * 0.6, 2)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bottom = np.zeros(len(rows))
    for c, col in zip(cats, ccol):
        v = t[c].to_numpy() if c in t else np.zeros(len(rows))
        ax.bar(x, v, bottom=bottom, color=col, width=0.85, edgecolor="white", linewidth=1, label=c)
        bottom += v
    for xi, tot in zip(x, bottom):
        ax.text(xi, tot + 0.8, f"{int(tot)}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, [s for _, s in rows], fontsize=7)
    for i, m in enumerate(METHODS):
        ax.text(x[2 * i:2 * i + 2].mean(), -9, m, ha="center", fontsize=8, color=INK)
    ax.set(ylabel="Failed frames (of 100)", ylim=(0, max(bottom) + 6))
    ax.legend(fontsize=7, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="x", visible=False)
    save(fig, "fig_failure_modes.png")


def fig_failure_cause():
    d = pd.read_csv(A / "diag_failure_cause.csv")
    d["label"] = d["scene"] + "\n" + d["obs"].map({"oracle": "SIL-oracle", "edges": "SIL-edge"})
    cats = [("search", "Search failure (truth fits better)", "#2a78d6"),
            ("tie", "Tie (within 5%)", "#c3c2b7"),
            ("objective", "Objective failure (wrong pose fits better)", "#eb6834")]
    fig, ax = plt.subplots(figsize=(5.5, 3))
    left = np.zeros(len(d))
    for c, lab, col in cats:
        ax.barh(d["label"], d[c], left=left, color=col, edgecolor="white", linewidth=1, label=lab, height=0.6)
        for i, (l, v) in enumerate(zip(left, d[c])):
            if v >= 3:
                ax.text(l + v / 2, i, str(v), ha="center", va="center", fontsize=7,
                        color="white" if col != "#c3c2b7" else INK)
        left += d[c].to_numpy()
    ax.invert_yaxis()
    ax.set(xlabel="Failed frames")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_failure_cause.png")


def fig_hybrid():
    h = pd.read_csv(A / "diag_hybrid.csv")
    h["pnp"] = h["method"].str.split("+").str[0]
    h["obs"] = h["method"].str.split("+").str[1]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    order = [(s, p) for s in ("v1", "v2") for p in ("ORB", "AKAZE", "LightGlue")]
    y = np.arange(len(order))
    for ax, (pcol, hcol, lab, fmt) in zip(axes, [("pnp_success_pct", "hybrid_success_pct", "Success (%)", "{:.0f}"),
                                                ("rot_med_pnp", "rot_med_hyb", "Median rotation error (°)", "{:.2f}")]):
        for off, obs, mk in ((-0.15, "SIL-oracle", "o"), (0.15, "SIL-edges", "D")):
            g = h[h["obs"] == obs].set_index(["scene", "pnp"]).reindex(order)
            for yi, (a, b) in enumerate(zip(g[pcol], g[hcol])):
                ax.plot([a, b], [yi + off] * 2, color=GRID, lw=2, zorder=1)
            ax.scatter(g[pcol], y + off, color=[COL[p] for _, p in order], marker="|", s=80, zorder=2)
            ax.scatter(g[hcol], y + off, color=[COL[p] for _, p in order], marker=mk, s=22, zorder=3,
                       edgecolors="white", linewidths=0.5)
            if ax is axes[0]:
                ax.scatter([], [], marker=mk, color=INK, s=22,
                           label=f"after {('oracle' if 'oracle' in obs else 'edge')} polish")
        ax.set_yticks(y, [f"{s} {p}" for s, p in order])
        ax.invert_yaxis()
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
    axes[0].scatter([], [], marker="|", color=INK, s=80, label="PnP alone")
    axes[0].legend(fontsize=7, loc="lower left")
    axes[1].set_yticklabels([])
    save(fig, "fig_hybrid.png")


def fig_symcheck():
    d = pd.read_csv(A / "diag_symcheck.csv")
    d["label"] = d["scene"] + " " + d["method"].str.replace("+sym", "", regex=False)
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    y = np.arange(len(d))
    ax.barh(y - 0.18, d["rescued"], height=0.34, color="#1baf7a", edgecolor="white", label="Rescued (fail → success)")
    ax.barh(y + 0.18, d["broken"], height=0.34, color="#eb6834", edgecolor="white", label="Broken (success → fail)")
    for yi, (r, b, p) in enumerate(zip(d["rescued"], d["broken"], d["mcnemar_p"])):
        ax.text(max(r, b) + 0.6, yi, f"p = {p:.3f}" if p >= 0.001 else "p < 0.001", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, d["label"])
    ax.invert_yaxis()
    ax.set(xlabel="Frames", xlim=(0, d[["rescued", "broken"]].max().max() + 7))
    ax.legend(fontsize=7, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="y", visible=False)
    save(fig, "fig_symcheck.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(A / "all_results_combined.csv")
    print("Figures ->", OUT)
    fig_cdf(df)
    fig_error_vs_range(df)
    fig_failure_modes(df)
    for fn in (fig_failure_cause, fig_hybrid, fig_symcheck):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"   (skipped {fn.__name__}: {e.filename} missing)")


if __name__ == "__main__":
    main()
