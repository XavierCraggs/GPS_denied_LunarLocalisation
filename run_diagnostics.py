"""Follow-up diagnostics for the silhouette method (run after the full v1/v2 runs).

  truth   Why did each silhouette failure happen?
          - Polish the GROUND-TRUTH pose with the fine levels and compare its cost with the cost the
            full run finished on. Truth clearly cheaper -> SEARCH failure (a better answer existed,
            the search never reached it). Answer clearly cheaper -> OBJECTIVE failure (the cost
            function itself prefers the wrong pose). Within 5% -> TIE (the image can't tell them apart).
          - Also polish the lander's 90/180/270 deg symmetric poses -> how ambiguous is the outline?
  symcheck Full localisation, then also polish the estimate's 90/180/270 deg alternatives and
          keep the lowest-cost pose. Fixes symmetric mix-ups the search makes?
  hybrid  Start the silhouette refinement from each PnP estimate instead of the template search.
          Does it improve PnP? Does the silhouette cost at the PnP pose flag PnP's gross errors?
  summary Tables for both (all scenes found) -> analysis/diag_*.csv

Usage (PowerShell, from the repo folder):
  $env:SCENE="v1"; python run_diagnostics.py truth            # add --frames 5 62 for a quick test
  $env:SCENE="v1"; python run_diagnostics.py hybrid           # needs a PnP CSV with est_rx/ry/rz
  python run_diagnostics.py summary
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu, wilcoxon

import PnP_Estimator as pe
import Silhouette_Estimator as se
import run_experiments as rx

HERE = Path(__file__).resolve().parent
FINE = se.LEVELS[1:]                                        # start is already close -> skip coarse level
FINAL_TAU, FINAL_TRIM = se.LEVELS[-1][0], se.LEVELS[-1][4]  # score poses exactly like the full run's final cost
SYM_ANGLES = (90, 180, 270)
SLID_BACK_DEG = 30          # a "symmetric" polish that ends this close to the truth didn't stay symmetric
TIE = 0.05                  # costs within 5% count as a tie
FLOOR = 0.05                # px added to both sides of every cost ratio (a perfect 0.00 fit would divide by zero)
OBS = ("oracle", "edges")
RENAME = {"orb": "ORB", "akaze": "AKAZE", "lightglue": "LightGlue"}


def _latest(pattern, required=True):
    hits = sorted(HERE.glob(pattern))
    if not hits and required:
        raise SystemExit(f"No file matches {pattern}")
    return hits[-1] if hits else None


def _ok(pos, rot):
    return (pos < pe.POS_MAX_ERR) & (rot < pe.ROT_MAX_ERR)


# ---------------------------------------------------------------- shared helpers (run inside workers)
def final_cost(R, C, obs):
    """Cost at a fixed pose, scored like the final refinement level (tau 10, trim 0.8)."""
    ctx = se.make_ctx(rx._W["V"], rx._W["F"], rx._W["K"], R, C, obs, FINAL_TAU)
    return se.cost(np.zeros(6), ctx, trim=FINAL_TRIM)


def polish(R, C, obs):
    """Fine levels only, from a pose that is already roughly right. Returns R, C, cost, nfev."""
    return se.refine(rx._W["V"], rx._W["F"], rx._W["K"], R, C, obs, levels=FINE)


def sym_pose(R, C, L, deg):
    """Camera pose that sees the lander exactly as if the lander were turned by deg about the
    vertical (world z) axis through its centre L. Same range and height, different azimuth."""
    Rz = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(deg)]))[0]
    return R @ Rz, L + Rz.T @ (C - L)


# ---------------------------------------------------------------- truth-cost diagnostic
def truth_task(frame, obs_name):
    gt = rx._W["poses"][frame]
    obs = rx._observations(frame, rx.FINAL_EDGE_CFG)[obs_name]
    L = rx._W["V"].mean(axis=0)
    R0, C0 = gt["R_world2cv"], gt["C"]
    t0 = time.perf_counter()

    row = dict(test_frame=frame, obs=obs_name, gt_range=gt["range"], gt_azimuth=gt["azimuth"],
               cost_gt_raw=final_cost(R0, C0, obs))                 # exactly at the truth
    R, C, row["cost_truth"], _ = polish(R0, C0, obs)                # best cost in the truth's basin
    row["truth_drift_m"], row["truth_drift_deg"] = se.pose_err(R, C, gt)
    for deg in SYM_ANGLES:
        R, C, row[f"cost_sym{deg}"], _ = polish(*sym_pose(R0, C0, L, deg), obs)
        row[f"sym{deg}_pos_err"], row[f"sym{deg}_rot_err"] = se.pose_err(R, C, gt)
    row["runtime_s"] = time.perf_counter() - t0
    return row


def exp_truth(frames):
    tasks = [dict(frame=f, obs_name=o) for f in frames for o in OBS]
    out = se.RESULTS_DIR / f"truthcost_{pe.SCENE}_{time.strftime('%Y%m%d_%H%M')}.csv"
    rx.run_parallel(truth_task, tasks, out,
                    lambda r: (f"f{r['test_frame']:3d} {r['obs']:6s} truth {r['cost_truth']:.2f} "
                               f"(drift {r['truth_drift_m']:.2f} m) | sym90 {r['cost_sym90']:.2f} "
                               f"sym180 {r['cost_sym180']:.2f} sym270 {r['cost_sym270']:.2f}  {r['runtime_s']:.0f}s"))
    print(f"-> {out}")


# ---------------------------------------------------------------- hybrid PnP -> silhouette
def hybrid_task(frame, pnp, obs_name):
    gt = rx._W["poses"][frame]
    row = dict(test_frame=frame, method=f"{pnp['method']}+SIL-{obs_name}", pnp_method=pnp["method"],
               obs=obs_name, gt_range=gt["range"], gt_azimuth=gt["azimuth"],
               pnp_pos_err=pnp["pos_err_m"], pnp_rot_err=pnp["rot_err_deg"])
    if not pnp["success"]:
        return dict(row, success=False)                              # no PnP pose -> nothing to refine
    R0 = cv2.Rodrigues(np.array([pnp["est_rx"], pnp["est_ry"], pnp["est_rz"]], float))[0]
    C0 = np.array([pnp["est_x"], pnp["est_y"], pnp["est_z"]], float)
    obs = rx._observations(frame, rx.FINAL_EDGE_CFG)[obs_name]

    t0 = time.perf_counter()
    cost_start = final_cost(R0, C0, obs)                             # the "verification" score
    R, C, fun, nfev = polish(R0, C0, obs)
    pos, rot = se.pose_err(R, C, gt)
    return dict(row, success=True, pos_err_m=pos, rot_err_deg=rot, est_x=C[0], est_y=C[1], est_z=C[2],
                cost_start=cost_start, cost=fun, nfev=nfev, runtime_s=time.perf_counter() - t0)


def exp_hybrid(frames):
    pnp_csv = _latest(f"results/pnp_results_{pe.SCENE}_*.csv")
    df = pd.read_csv(pnp_csv)
    if "est_rx" not in df:
        raise SystemExit(f"{pnp_csv.name} has no est_rx/ry/rz -> re-run PnP_Estimator.py (with the rotation columns) first")
    print(f"PnP starts <- {pnp_csv.name}")
    df["method"] = df["method"].replace(RENAME)
    df["success"] = df["success"].astype(str).str.lower() == "true"
    pnp_rows = df[df["test_frame"].isin(frames)].to_dict("records")
    tasks = [dict(frame=int(p["test_frame"]), pnp=p, obs_name=o) for p in pnp_rows for o in OBS]
    out = se.RESULTS_DIR / f"hybrid_{pe.SCENE}_{time.strftime('%Y%m%d_%H%M')}.csv"

    def describe(r):
        if not r["success"]:
            return f"f{r['test_frame']:3d} {r['method']:20s} (PnP gave no pose)"
        return (f"f{r['test_frame']:3d} {r['method']:20s} pos {r['pnp_pos_err']:.3f} -> {r['pos_err_m']:.3f} m  "
                f"rot {r['pnp_rot_err']:.2f} -> {r['rot_err_deg']:.2f}°  start cost {r['cost_start']:.2f}  {r['runtime_s']:.0f}s")
    rx.run_parallel(hybrid_task, tasks, out, describe)
    print(f"-> {out}")


# ---------------------------------------------------------------- symmetry check (full localisation + 3 alternatives)
def symcheck_task(frame, method):
    """Full silhouette localisation (as in the full run), then refine the estimate turned 90/180/270 deg
    about the lander's vertical axis (full coarse-to-fine schedule) and keep whichever pose has the
    lowest final cost. No tuned parameters."""
    gt = rx._W["poses"][frame]
    obs = rx._observations(frame, rx.FINAL_EDGE_CFG)["oracle" if method == "SIL-oracle" else "edges"]
    V, F, K = rx._W["V"], rx._W["F"], rx._W["K"]
    L = V.mean(axis=0)
    t0 = time.perf_counter()
    R, C, fun, info = se.localise_silhouette(V, F, K, rx._W["ref_db"], obs)
    base_s = time.perf_counter() - t0
    base_pos, base_rot, base_cost = *se.pose_err(R, C, gt), fun

    best, costs = (R, C, fun, 0), {}
    for deg in SYM_ANGLES:
        Rs, Cs, fs, _ = se.refine(V, F, K, *sym_pose(R, C, L, deg), obs)   # all levels: alt start can be ~2-3 m off
        costs[f"cost_alt{deg}"] = fs
        if fs < best[2]:
            best = (Rs, Cs, fs, deg)
    R, C, fun, deg = best
    pos, rot = se.pose_err(R, C, gt)
    return dict(test_frame=frame, method=f"{method}+sym", success=True, ref_frame=info["ref_frame"],
                gt_range=gt["range"], gt_azimuth=gt["azimuth"], pos_err_m=pos, rot_err_deg=rot,
                est_x=C[0], est_y=C[1], est_z=C[2], runtime_s=time.perf_counter() - t0, cost=fun,
                chosen_deg=deg, base_pos_err=base_pos, base_rot_err=base_rot, base_cost=base_cost,
                base_runtime_s=base_s, **costs)


def exp_symcheck(frames):
    tasks = [dict(frame=f, method=m) for f in frames for m in ("SIL-oracle", "SIL-edge")]
    out = se.RESULTS_DIR / f"symcheck_{pe.SCENE}_{time.strftime('%Y%m%d_%H%M')}.csv"
    rx.run_parallel(symcheck_task, tasks, out,
                    lambda r: (f"f{r['test_frame']:3d} {r['method']:14s} base {r['base_pos_err']:.3f} m {r['base_rot_err']:.1f}° "
                               f"-> alt {r['chosen_deg']:3d}: {r['pos_err_m']:.3f} m {r['rot_err_deg']:.1f}°  {r['runtime_s']:.0f}s"))
    print(f"-> {out}")


# ---------------------------------------------------------------- summaries (main process only)
def truth_table(scene):
    t_csv = _latest(f"results_silhouette/truthcost_{scene}_*.csv", required=False)
    s_csv = _latest(f"results_silhouette/sil_results_{scene}_*.csv", required=False)
    if t_csv is None or s_csv is None:
        return None
    t, s = pd.read_csv(t_csv), pd.read_csv(s_csv)
    s["obs"] = s["method"].map({"SIL-oracle": "oracle", "SIL-edge": "edges"})
    s["ok"] = _ok(s["pos_err_m"], s["rot_err_deg"])
    d = t.merge(s[["test_frame", "obs", "ok", "cost", "pos_err_m", "rot_err_deg"]], on=["test_frame", "obs"])
    d["scene"] = scene

    # a symmetric polish that slid back to the truth says nothing about ambiguity -> ignore it
    for a in SYM_ANGLES:
        d.loc[d[f"sym{a}_rot_err"] < SLID_BACK_DEG, f"cost_sym{a}"] = np.nan
    ratio = lambda c: (c + FLOOR) / (d["cost_truth"] + FLOOR)
    d["ratio_180"] = ratio(d["cost_sym180"])
    d["ratio_90"] = ratio(d[["cost_sym90", "cost_sym270"]].min(axis=1))

    # failure cause: compare the truth basin's best cost with the cost the full run ended on
    r = ratio(d["cost"])                                              # >1 -> the truth fits better
    d["cause"] = np.select([d["ok"], r > 1 + TIE, r < 1 - TIE], ["ok", "search", "objective"], default="tie")
    return d


def summarise_truth():
    d = pd.concat([x for x in (truth_table(s) for s in ("v1", "v2")) if x is not None], ignore_index=True)
    causes = d.groupby(["scene", "obs", "cause"]).size().unstack("cause", fill_value=0)
    amb = d.groupby(["scene", "obs"]).agg(
        n=("test_frame", "size"),
        flip180_tie_pct=("ratio_180", lambda x: (x < 1 + TIE).mean() * 100),   # 180 pose fits as well as truth
        alias90_tie_pct=("ratio_90", lambda x: (x < 1 + TIE).mean() * 100),
        ratio180_med=("ratio_180", "median"), ratio90_med=("ratio_90", "median"),
        truth_drift_cm_med=("truth_drift_m", lambda x: x.median() * 100)).round(2)
    print("\n=== silhouette failure cause (search = truth fits >5% better than the answer) ===")
    print(causes.to_string())
    print("\n=== symmetry ambiguity (ratio = symmetric-pose cost / truth cost; <1.05 = can't tell apart) ===")
    print(amb.to_string())
    return {"diag_truth_frames": d, "diag_failure_cause": causes, "diag_symmetry": amb}


def hybrid_table(scene):
    h_csv = _latest(f"results_silhouette/hybrid_{scene}_*.csv", required=False)
    if h_csv is None:
        return None
    h = pd.read_csv(h_csv)
    h["scene"] = scene
    h["success"] = h["success"].astype(str).str.lower() == "true"
    h["pnp_ok"] = _ok(h["pnp_pos_err"].fillna(np.inf), h["pnp_rot_err"].fillna(np.inf))
    h["ok"] = h["success"] & _ok(h["pos_err_m"].fillna(np.inf), h["rot_err_deg"].fillna(np.inf))
    return h


def summarise_hybrid():
    h = pd.concat([x for x in (hybrid_table(s) for s in ("v1", "v2")) if x is not None], ignore_index=True)
    rows = []
    for (scene, method), g in h.groupby(["scene", "method"]):
        both = g[g["pnp_ok"] & g["ok"]]
        p = lambda a, b: wilcoxon(a, b).pvalue if len(both) >= 10 else np.nan
        posed = g[g["success"]]
        bad, good = posed.loc[~posed["pnp_ok"], "cost_start"], posed.loc[posed["pnp_ok"], "cost_start"]
        auc = (mannwhitneyu(bad, good).statistic / (len(bad) * len(good))) if len(bad) and len(good) else np.nan
        rows.append(dict(scene=scene, method=method,
                         pnp_success_pct=g["pnp_ok"].mean() * 100, hybrid_success_pct=g["ok"].mean() * 100,
                         rescued=int((~g["pnp_ok"] & g["ok"]).sum()), broken=int((g["pnp_ok"] & ~g["ok"]).sum()),
                         n_both=len(both),
                         pos_med_cm_pnp=both["pnp_pos_err"].median() * 100, pos_med_cm_hyb=both["pos_err_m"].median() * 100,
                         pos_p=p(both["pnp_pos_err"], both["pos_err_m"]),
                         rot_med_pnp=both["pnp_rot_err"].median(), rot_med_hyb=both["rot_err_deg"].median(),
                         rot_p=p(both["pnp_rot_err"], both["rot_err_deg"]),
                         n_pnp_bad=len(bad), flag_auc=auc,      # 1.0 = start cost perfectly separates bad PnP poses
                         start_cost_med_good=good.median(), start_cost_med_bad=bad.median()))
    t = pd.DataFrame(rows).set_index(["scene", "method"]).round(4)
    print("\n=== hybrid: PnP start -> silhouette polish (accuracy on frames both got right) ===")
    print(t.to_string())
    return {"diag_hybrid_frames": h, "diag_hybrid": t}


def summarise_symcheck():
    rows = []
    for scene in ("v1", "v2"):
        csv = _latest(f"results_silhouette/symcheck_{scene}_*.csv", required=False)
        if csv is None:
            continue
        d = pd.read_csv(csv)
        d["base_ok"], d["ok"] = _ok(d["base_pos_err"], d["base_rot_err"]), _ok(d["pos_err_m"], d["rot_err_deg"])
        for method, g in d.groupby("method"):
            gained, lost = int((~g["base_ok"] & g["ok"]).sum()), int((g["base_ok"] & ~g["ok"]).sum())
            rows.append(dict(scene=scene, method=method, n=len(g),
                             base_success_pct=g["base_ok"].mean() * 100, sym_success_pct=g["ok"].mean() * 100,
                             rescued=gained, broken=lost,
                             mcnemar_p=binomtest(gained, gained + lost).pvalue if gained + lost else 1.0,
                             switched=int((g["chosen_deg"] != 0).sum()),
                             runtime_base_s=g["base_runtime_s"].median(), runtime_sym_s=g["runtime_s"].median()))
    if not rows:
        raise ValueError("no symcheck results")
    t = pd.DataFrame(rows).set_index(["scene", "method"]).round(4)
    print("\n=== symmetry check: full localisation vs + best of 90/180/270 alternatives ===")
    print(t.to_string())
    return {"diag_symcheck": t}


def summary():
    out = HERE / "analysis"
    out.mkdir(exist_ok=True)
    tables = {}
    for fn in (summarise_truth, summarise_hybrid, summarise_symcheck):
        try:
            tables.update(fn())
        except ValueError:                       # pd.concat([]) -> nothing run yet for this diagnostic
            print(f"\n({fn.__name__}: no results yet)")
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=not name.endswith("frames"))
    print(f"\nTables -> {out}")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Silhouette diagnostics")
    ap.add_argument("experiment", choices=["truth", "hybrid", "symcheck", "summary"])
    ap.add_argument("--frames", type=int, nargs="+", help="only these test frames (quick test)")
    args = ap.parse_args()
    if args.experiment == "summary":
        summary()
    else:
        se.RESULTS_DIR.mkdir(exist_ok=True)
        frames = args.frames or sorted(pe.load_poses(pe.TEST_DIR / "posev_testing.csv"))
        {"truth": exp_truth, "hybrid": exp_hybrid, "symcheck": exp_symcheck}[args.experiment](frames)