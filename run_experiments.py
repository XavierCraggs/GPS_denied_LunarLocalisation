""" Experiments driver for the Silhouette method. Experiments split into seperate functions"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd

import Silhouette_Estimator as se
import PnP_Estimator as pe

SEED = 0
_W = {}


CLAHE_CFG = dict(clahe_clip=2.0, blur_sigma=1.2, canny_low=10, canny_high=30)
FINAL_EDGE_CFG = CLAHE_CFG     

#------------

def _init_worker():
    """ Runs once for each worker process to remove compute overhead """
    cv2.setNumThreads(1)
    mesh = se.SIL_MESH_PATH if se.SIL_MESH_PATH.exists() else se.MESH_PATH
    _W["V"], _W["F"] = se.load_mesh(mesh)
    _W["K"] = pe.load_intrinsics(pe.INTRINSICS_PATH)
    _W["poses"] = pe.load_poses(pe.TEST_DIR / "posev_testing.csv")
    _W["ref_db"] = se.build_ref_db(_W["V"], _W["F"], _W["K"], pe.load_poses(pe.REF_DIR / "posev_training.csv"))

def _observations(frame, edge_cfg):
    """Oracle + real edge maps for one test frame."""
    img = cv2.imread(str(pe.TEST_DIR / "RGB" / f"frame_{frame:04d}.png"))
    _, index_map = pe.decompose_EXR(pe.TEST_DIR / "EXR" / f"frame_{frame:04d}.exr")
    return {"oracle": se.oracle_edges(index_map == pe.LANDER_INDEX),
            "edges": se.edge_map(img, **edge_cfg)}


def refine_task(frame, noise_idx, rot_deg, pos_m, variant):
    """One refine from a perturbed GT start. Same (frame, noise_idx) -> same start in every variant."""
    gt = _W["poses"][frame]
    rng = np.random.default_rng([SEED, frame, noise_idx])          # deterministic, process-independent
    R0, C0 = se.add_noise(gt["R_world2cv"], gt["C"], rot_deg, pos_m, rng)
    obs = _observations(frame, variant.get("edge_cfg", {}))[variant["obs"]]

    t0 = time.perf_counter()
    R, C, fun, nfev = se.refine(_W["V"], _W["F"], _W["K"], R0, C0, obs,
                                lam=variant.get("lam", 0.0), levels=variant.get("levels", se.LEVELS))
    pos, rot = se.pose_err(R, C, gt)
    return dict(frame=frame, range=gt["range"], noise_rot=rot_deg, noise_pos=pos_m,
                variant=variant["name"], obs=variant["obs"], pos_err=pos, rot_err=rot,
                ok=bool(pos < pe.POS_MAX_ERR and rot < pe.ROT_MAX_ERR),
                cost=fun, nfev=nfev, runtime=time.perf_counter() - t0)


# ---------------------------------------------------------------- main side
def run_parallel(fn, tasks, out_csv, describe, workers=None):
    """tasks: list of kwargs dicts for refine_task. Saves the CSV after every result."""
    workers = workers or max(1, os.cpu_count() - 2)
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = [pool.submit(fn, **t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            rows.append(r)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"[{i:3d}/{len(tasks)}] {describe(r)}  ({time.perf_counter() - t0:.0f}s)")
    return pd.DataFrame(rows)

def describe_refine(r):
    return (f"f{r['frame']:3d} {r['noise_rot']:>2}°/{r['noise_pos']}m {r['variant']:22s} "
            f"pos {r['pos_err']:.3f} m rot {r['rot_err']:.2f}° {'OK' if r['ok'] else '--'}  cost {r['cost']:.2f}")


def summary(df):
    print(df.groupby(["variant", "noise_rot"])
            .agg(success=("ok", "mean"), pos_med=("pos_err", "median"),
                 rot_med=("rot_err", "median"), cost_med=("cost", "median"), n=("ok", "size"))
            .round(3).to_string())


def test_frames(step=7):
    poses = pe.load_poses(pe.TEST_DIR / "posev_testing.csv")
    return sorted(poses, key=lambda f: poses[f]["range"])[::step]        # ~15 frames spread by range


def build_tasks(frames, noise, variants):
    return [dict(frame=f, noise_idx=i, rot_deg=r, pos_m=p, variant=v)
            for f in frames for i, (r, p) in enumerate(noise) for v in variants]


# ---------------------------------------------------------------- experiments
def exp_basin():
    """Convergence basin: success vs starting error, oracle vs edges."""
    noise = [(0, 0.0), (2, 0.5), (5, 1.0), (10, 2.0)]
    variants = [dict(name="oracle", obs="oracle"), dict(name="edges", obs="edges")]
    out = se.RESULTS_DIR / f"basin_{pe.SCENE}_{time.strftime('%Y%m%d_%H%M')}.csv"
    summary(run_parallel(refine_task, build_tasks(test_frames(), noise, variants), out, describe_refine))


def exp_diagnose():
    """Why do edges fail? Zero-noise control + one round of candidate fixes."""
    noise = [(0, 0.0), (2, 0.5)]
    clahe = dict(clahe_clip=2.0, blur_sigma=1.2, canny_low=10, canny_high=30)
    variants = [
        dict(name="edges base", obs="edges"),
        dict(name="edges lam0.5", obs="edges", lam=0.5),
        dict(name="edges trim0.8", obs="edges", levels=tuple(l[:4] + (0.8,) for l in se.LEVELS)),
        dict(name="edges CLAHE", obs="edges", edge_cfg=clahe),
    ]
    out = se.RESULTS_DIR / f"diagnose_{pe.SCENE}_{time.strftime('%Y%m%d_%H%M')}.csv"
    summary(run_parallel(refine_task, build_tasks(test_frames(), noise, variants), out, describe_refine))


def exp_schedule():
    """Final tuning round: trimming only at the finest level, with and without CLAHE."""
    noise = [(0, 0.0), (2, 0.5)]
    untrimmed = tuple(l[:4] + (1.0,) for l in se.LEVELS)
    variants = [
        dict(name="edges untrimmed", obs="edges", levels=untrimmed),
        dict(name="edges trim fine", obs="edges"),
        dict(name="edges CLAHE trim fine", obs="edges", edge_cfg=CLAHE_CFG),
        dict(name="oracle trim fine", obs="oracle"),
    ]
    out = se.RESULTS_DIR / f"schedule_{pe.SCENE}_{time.strftime('%Y%m%d_%H%M')}.csv"
    summary(run_parallel(refine_task, build_tasks(test_frames(), noise, variants), out, describe_refine))

def exp_init():
    """Global-search quality: start-pose error per frame + overlays (green = GT, red = start)."""
    _init_worker()
    out_dir = pe.DEBUG_DIR / "init"
    out_dir.mkdir(parents=True, exist_ok=True)
    V, F, K, L = _W["V"], _W["F"], _W["K"], _W["V"].mean(axis=0)
    rows = []
    for frame in test_frames():
        gt = _W["poses"][frame]
        img = cv2.imread(str(pe.TEST_DIR / "RGB" / f"frame_{frame:04d}.png"))
        for obs_name, obs_edges in _observations(frame, FINAL_EDGE_CFG).items():
            t0 = time.perf_counter()
            cand = se.search_refs(obs_edges, _W["ref_db"])[0]
            R0, C0 = se.start_pose(cand, K, L)
            pos, rot = se.pose_err(R0, C0, gt)
            rows.append(dict(frame=frame, range=gt["range"], obs=obs_name, ref=cand["ref"]["frame"],
                             scale=cand["scale"], start_pos_err=pos, start_rot_err=rot,
                             search_s=time.perf_counter() - t0))
            print(f"f{frame:3d} {gt['range']:5.1f} m {obs_name:6s} ref {cand['ref']['frame']:2d} "
                  f"s {cand['scale']:.2f}: start {pos:.2f} m {rot:.1f}°")
            if obs_name == "edges":
                vis = se.draw_overlay(img, se.project_mask(V, F, gt["R_world2cv"], -gt["R_world2cv"] @ gt["C"], K))
                vis = se.draw_overlay(vis, se.project_mask(V, F, R0, -R0 @ C0, K), (0, 0, 255))
                cv2.imwrite(str(out_dir / f"{pe.SCENE}_init_{frame:04d}.png"), vis)
    pd.DataFrame(rows).to_csv(se.RESULTS_DIR / f"init_{pe.SCENE}_{time.strftime('%Y%m%d_%H%M')}.csv", index=False)


def exp_full():
    """All 100 test frames x {SIL-oracle, SIL-edge} -> results in the PnP CSV format."""
    frames = sorted(pe.load_poses(pe.TEST_DIR / "posev_testing.csv"))
    tasks = [dict(frame=f, method=m) for f in frames for m in ("SIL-oracle", "SIL-edge")]
    out = se.RESULTS_DIR / f"sil_results_{pe.SCENE}_{time.strftime('%Y%m%d_%H%M')}.csv"
    df = run_parallel(full_task, tasks, out,
                      lambda r: f"f{r['test_frame']:3d} {r['method']:10s} pos {r['pos_err_m']:.3f} m "
                                f"rot {r['rot_err_deg']:.2f}° (start {r['start_pos_err']:.2f} m) {r['runtime_s']:.0f}s")
    pe.summarise(df)

def full_task(frame, method):
    """Complete silhouette localisation of one test frame, as a PnP-format results row."""
    gt = _W["poses"][frame]
    obs = _observations(frame, FINAL_EDGE_CFG)["oracle" if method == "SIL-oracle" else "edges"]
    t0 = time.perf_counter()
    R, C, fun, info = se.localise_silhouette(_W["V"], _W["F"], _W["K"], _W["ref_db"], obs)
    runtime = time.perf_counter() - t0
    pos, rot = se.pose_err(R, C, gt)
    s_pos, s_rot = se.pose_err(*info["start"], gt)
    return dict(test_frame=frame, method=method, success=True, ref_frame=info["ref_frame"],
                gt_range=gt["range"], gt_azimuth=gt["azimuth"], pos_err_m=pos, rot_err_deg=rot,
                est_x=C[0], est_y=C[1], est_z=C[2], runtime_s=runtime, cost=fun,
                scale=info["scale"], start_pos_err=s_pos, start_rot_err=s_rot, nfev=info["nfev"])


EXPERIMENTS = {"basin": exp_basin, "diagnose": exp_diagnose, "schedule": exp_schedule,
               "init": exp_init, "full": exp_full}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Silhouette experiments")
    ap.add_argument("experiment", choices=EXPERIMENTS)
    se.RESULTS_DIR.mkdir(exist_ok=True)
    EXPERIMENTS[ap.parse_args().experiment]()