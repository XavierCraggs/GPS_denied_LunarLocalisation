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

#------------

def _init_worker():
    """ Runs once for each worker process to remove compute overhead """
    cv2.setNumThreads(1)
    mesh = se.SIL_MESH_PATH if se.SIL_MESH_PATH.exists else se.MESH_PATH
    _W["V"], _W["F"] = se.load_mesh(mesh)
    _W["K"] = pe.load_intrinsics(pe.INTRINSICS_PATH)
    _W["poses"] = pe.load_poses(pe.TEST_DIR / "posev_testing.csv")

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
                                lam=variant.get("lam", 0.0), trim=variant.get("trim", 1.0))
    pos, rot = se.pose_err(R, C, gt)
    return dict(frame=frame, range=gt["range"], noise_rot=rot_deg, noise_pos=pos_m,
                variant=variant["name"], obs=variant["obs"], pos_err=pos, rot_err=rot,
                ok=bool(pos < pe.POS_MAX_ERR and rot < pe.ROT_MAX_ERR),
                cost=fun, nfev=nfev, runtime=time.perf_counter() - t0)


# ---------------------------------------------------------------- main side
def run_parallel(tasks, out_csv, workers=None):
    """tasks: list of kwargs dicts for refine_task. Saves the CSV after every result."""
    workers = workers or max(1, os.cpu_count() - 2)
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = [pool.submit(refine_task, **t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            rows.append(r)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"[{i:3d}/{len(tasks)}] f{r['frame']:3d} {r['noise_rot']:>2}°/{r['noise_pos']}m "
                  f"{r['variant']:14s} pos {r['pos_err']:.3f} m rot {r['rot_err']:.2f}° "
                  f"{'OK' if r['ok'] else '--'}  cost {r['cost']:.2f}  ({time.perf_counter() - t0:.0f}s)")
    return pd.DataFrame(rows)


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
    summary(run_parallel(build_tasks(test_frames(), noise, variants), out))


def exp_diagnose():
    """Why do edges fail? Zero-noise control + one round of candidate fixes."""
    noise = [(0, 0.0), (2, 0.5)]
    clahe = dict(clahe_clip=2.0, blur_sigma=1.2, canny_low=10, canny_high=30)
    variants = [
        dict(name="edges base", obs="edges"),
        dict(name="edges lam0.5", obs="edges", lam=0.5),
        dict(name="edges trim0.8", obs="edges", trim=0.8),
        dict(name="edges CLAHE", obs="edges", edge_cfg=clahe),
    ]
    out = se.RESULTS_DIR / f"diagnose_{pe.SCENE}_{time.strftime('%Y%m%d_%H%M')}.csv"
    summary(run_parallel(build_tasks(test_frames(), noise, variants), out))


EXPERIMENTS = {"basin": exp_basin, "diagnose": exp_diagnose}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Silhouette experiments")
    ap.add_argument("experiment", choices=EXPERIMENTS)
    se.RESULTS_DIR.mkdir(exist_ok=True)
    EXPERIMENTS[ap.parse_args().experiment]()