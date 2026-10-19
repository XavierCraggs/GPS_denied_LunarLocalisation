"""Results analysis for the thesis: every table and figure from the raw result CSVs.

Usage:  python analyse_results.py writes everything to ./analysis/
Inputs: results/pnp_results_*.csv (PnP), results_silhouette/sil_results_*.csv (silhouette).
"""
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                   # write files, never open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

import PnP_Estimator as pe
import Silhouette_Estimator as se

HERE = Path(__file__).resolve().parent
OUT = HERE / "analysis"
POS_OK, ROT_OK = pe.POS_MAX_ERR, pe.ROT_MAX_ERR         # one success definition everywhere
RANGE_BINS, RANGE_LABELS = [0, 11.5, 18.5, 30], ["near", "mid", "far"]
METHOD_ORDER = ["ORB", "AKAZE", "LightGlue", "SIL-edge", "SIL-oracle"]
RENAME = {"orb": "ORB", "akaze": "AKAZE", "lightglue": "LightGlue"}

# Which CSV backs each scene. v1 PnP predates the scene-tagged filenames, so it is named explicitly.
RUNS = {
    "v1": {"pnp": "results/pnp_results_v1_*.csv", "sil": "results_silhouette/sil_results_v1_*.csv"},
    "v2": {"pnp": "results/pnp_results_v2_*.csv",            "sil": "results_silhouette/sil_results_v2_*.csv"},
}


# ------------------------------------------------------------------ loading
def _latest(pattern):
    hits = sorted(HERE.glob(pattern))
    return hits[-1] if hits else None


def _scene_geometry(scene):
    """GT camera centres + lander centre for one scene (needed for the depth/lateral split)."""
    root = HERE.parent / f"render_output_{scene}" / "TESTING" / "posev_testing.csv"
    poses = pe.load_poses(root) if root.exists() else {}
    mesh = HERE / "Project_Intrinsics" / f"LAND_{scene}_sil.obj"
    L = se.load_mesh(mesh)[0].mean(axis=0) if mesh.exists() else None
    return poses, L


def load_all():
    frames = []
    for scene, files in RUNS.items():
        for kind, pattern in files.items():
            path = _latest(pattern)
            if path is None:
                print(f"  (no {kind} results for {scene} yet: {pattern})")
                continue
            df = pd.read_csv(path)
            df["method"] = df["method"].replace(RENAME)
            df["scene"], df["source"] = scene, path.name
            frames.append(df)
            print(f"  {scene} {kind:3s} <- {path.name} ({len(df)} rows)")
    df = pd.concat(frames, ignore_index=True)

    df["pose"] = df["success"].astype(str).str.lower() == "true"
    df["ok"] = df["pose"] & (df["pos_err_m"] < POS_OK) & (df["rot_err_deg"] < ROT_OK)
    df["range_bin"] = pd.cut(df["gt_range"], RANGE_BINS, labels=RANGE_LABELS)
    df["failure"] = np.select(
        [df["ok"], ~df["pose"], df["rot_err_deg"] > 150, df["rot_err_deg"] > 60, df["pos_err_m"] <= 2.0],
        ["ok", "no pose", "flip ~180°", "alias ~90°", "near miss (≤2 m)"], default="gross error")
    return _depth_lateral(df)


def _depth_lateral(df):
    """Split the position error into along-line-of-sight (depth) and perpendicular (lateral)."""
    df["depth_err_m"], df["lateral_err_m"] = np.nan, np.nan
    for scene in df["scene"].unique():
        poses, L = _scene_geometry(scene)
        if L is None or not poses:
            continue
        idx = df.index[(df["scene"] == scene) & df["pose"]]
        C_gt = np.array([poses[f]["C"] for f in df.loc[idx, "test_frame"]])
        C_est = df.loc[idx, ["est_x", "est_y", "est_z"]].to_numpy(float)
        los = (L - C_gt) / np.linalg.norm(L - C_gt, axis=1, keepdims=True)
        d = C_est - C_gt
        depth = np.einsum("ij,ij->i", d, los)
        df.loc[idx, "depth_err_m"] = np.abs(depth)
        df.loc[idx, "lateral_err_m"] = np.linalg.norm(d - depth[:, None] * los, axis=1)
    return df


# ------------------------------------------------------------------ tables
def _order(index):
    return [m for m in METHOD_ORDER if m in index]


def headline(df):
    ok = df[df["ok"]]
    g, gok = df.groupby(["scene", "method"]), ok.groupby(["scene", "method"])
    t = pd.DataFrame({
        "success_%":    g["ok"].mean() * 100,
        "failures":     g["ok"].apply(lambda s: int((~s).sum())),
        "pos_med_cm":   gok["pos_err_m"].median() * 100,
        "pos_p90_cm":   gok["pos_err_m"].quantile(0.9) * 100,
        "depth_med_cm": gok["depth_err_m"].median() * 100,
        "lat_med_cm":   gok["lateral_err_m"].median() * 100,
        "rot_med_deg":  gok["rot_err_deg"].median(),
        "runtime_s":    g["runtime_s"].median(),
    })
    return t.round(2)


def by_range(df):
    ok = df[df["ok"]]
    return (ok.groupby(["scene", "range_bin", "method"], observed=True)["pos_err_m"].median() * 100
            ).unstack("method").round(1)


def thresholds(df, levels=(0.25, 0.5, 1.0, 2.0)):
    rows = {}
    for p in levels:
        hit = df["pose"] & (df["pos_err_m"] < p) & (df["rot_err_deg"] < ROT_OK)
        rows[f"<{p} m"] = hit.groupby([df["scene"], df["method"]]).mean() * 100
    return pd.DataFrame(rows).round(1)


def taxonomy(df):
    return df.groupby(["scene", "method", "failure"]).size().unstack("failure", fill_value=0)


def silhouette_failure_source(df):
    """Silhouette failures: did the global search start badly (init) or did refinement fail?"""
    sil = df[df["method"].str.startswith("SIL") & ~df["ok"]].copy()
    if sil.empty:
        return pd.DataFrame()
    sil["cause"] = np.where(sil["start_pos_err"] > 2.0, "bad start (search)", "refinement")
    return sil.groupby(["scene", "method", "cause"]).size().unstack("cause", fill_value=0)


def paired_stats(df):
    """Same test frames across methods (and scenes) -> paired tests.
    McNemar exact: binomial test on frames where exactly one of the pair succeeds.
    Wilcoxon signed-rank: position error on frames both methods got right."""
    rows = []
    def pair(label, a, b):
        m = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        only_a = int((m["ok_a"] & ~m["ok_b"]).sum())
        only_b = int((~m["ok_a"] & m["ok_b"]).sum())
        p_mc = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
        both = m[m["ok_a"] & m["ok_b"]]
        p_w = wilcoxon(both["pos_err_m_a"], both["pos_err_m_b"]).pvalue if len(both) >= 10 else np.nan
        rows.append(dict(comparison=label, n=len(m), a_only_ok=only_a, b_only_ok=only_b, mcnemar_p=p_mc,
                         n_both_ok=len(both), med_a_cm=both["pos_err_m_a"].median() * 100,
                         med_b_cm=both["pos_err_m_b"].median() * 100, wilcoxon_p=p_w))
    key = lambda d: d.set_index("test_frame")[["ok", "pos_err_m"]]
    for scene in sorted(df["scene"].unique()):
        s = df[df["scene"] == scene]
        for a, b in combinations(_order(s["method"].unique()), 2):
            pair(f"{scene}: {a} vs {b}", key(s[s["method"] == a]), key(s[s["method"] == b]))
    if {"v1", "v2"} <= set(df["scene"]):
        for m in _order(df["method"].unique()):
            a, b = df[(df["scene"] == "v1") & (df["method"] == m)], df[(df["scene"] == "v2") & (df["method"] == m)]
            if len(a) and len(b):
                pair(f"{m}: v1 vs v2", key(a), key(b))
    return pd.DataFrame(rows).round(4)


# ------------------------------------------------------------------ figures
def fig_cdf(df):
    for scene in sorted(df["scene"].unique()):
        s = df[df["scene"] == scene]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for m in _order(s["method"].unique()):
            e = np.sort(s.loc[(s["method"] == m) & s["pose"], "pos_err_m"].to_numpy())
            n = (s["method"] == m).sum()                     # no-pose frames never reach the curve
            ax.step(e, np.arange(1, len(e) + 1) / n * 100, where="post", label=m)
        ax.axvline(POS_OK, color="k", lw=0.6, ls="--")
        ax.set(xscale="log", xlim=(1e-3, 60), ylim=(0, 100), xlabel="position error (m, log)",
               ylabel="frames with error ≤ x (%)", title=f"Cumulative position error ({scene})")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / f"cdf_pos_{scene}.png", dpi=150); plt.close(fig)


def fig_error_vs_range(df):
    for scene in sorted(df["scene"].unique()):
        s = df[(df["scene"] == scene) & df["ok"]]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for m in _order(s["method"].unique()):
            d = s[s["method"] == m]
            ax.scatter(d["gt_range"], d["pos_err_m"] * 100, s=10, label=m, alpha=0.7)
        ax.set(yscale="log", xlabel="range to lander (m)", ylabel="position error (cm, log)",
               title=f"Error vs range, successful frames ({scene})")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / f"error_vs_range_{scene}.png", dpi=150); plt.close(fig)


def fig_failures(tax):
    ax = tax.drop(columns="ok", errors="ignore").plot.bar(stacked=True, figsize=(8, 4))
    ax.set(ylabel="frames", title="Failure modes by method and scene")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(OUT / "failure_modes.png", dpi=150); plt.close()


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(exist_ok=True)
    print("Loading results:")
    df = load_all()
    df.to_csv(OUT / "all_results_combined.csv", index=False)

    tables = {"headline": headline(df), "by_range_median_cm": by_range(df), "threshold_success": thresholds(df),
              "failure_taxonomy": taxonomy(df), "silhouette_failure_source": silhouette_failure_source(df),
              "paired_stats": paired_stats(df)}
    for name, t in tables.items():
        t.to_csv(OUT / f"{name}.csv")
        print(f"\n=== {name} ===\n{t.to_string()}")

    fig_cdf(df)
    fig_error_vs_range(df)
    fig_failures(tables["failure_taxonomy"])
    print(f"\nTables + figures -> {OUT}")


if __name__ == "__main__":
    main()