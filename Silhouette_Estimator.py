import PnP_Estimator as pe
import numpy as np
from pathlib import Path
import cv2
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize



MESH_PATH = Path(__file__).resolve().parent / "Project_Intrinsics" / f"LAND_{pe.SCENE}.obj"
SIL_MESH_PATH = Path(__file__).resolve().parent / "Project_Intrinsics" / f"LAND_{pe.SCENE}_sil.obj"
RESULTS_DIR = Path(__file__).resolve().parent / "results_silhouette"

CONFIGS = {
    "report (CLAHE, 50/150)": dict(clahe_clip=2.0, blur_sigma=1.0, canny_low=50, canny_high=150),
    "no CLAHE, s1.2, 10/30":  dict(clahe_clip=None, blur_sigma=1.2, canny_low=10, canny_high=30),
    "CLAHE, s1.2, 10/30":     dict(clahe_clip=2.0, blur_sigma=1.2, canny_low=10, canny_high=30),
    "no CLAHE, s2.0, 20/50":  dict(clahe_clip=None, blur_sigma=2.0, canny_low=20, canny_high=50),
}

SLICE_NAMES = ["tilt dθx (°)", "pan dθy (°)", "roll dθz (°)",
               "right dCx (m)", "down dCy (m)", "forward dCz (m)"]

def load_mesh(path):
    V = []
    F = []  

    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:]])

    V = np.array(V, dtype=np.float32)
    F = np.array(F, dtype=np.int32)
    return V, F

def project_points(V, R, t, K):
    """ Converts 3D world points to 2D pixel coordinates and camera depth"""
    Xc = V @ R.T + t
    z = Xc[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = Xc[:, :2] / z[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, z

def project_mask(V, F, R, t, K, shape=(960, 1280), shift=2):
    """ Filled silhouette of the mesh at (R,t)"""
    uv, z = project_points(V, R, t, K)
    tri_uv, tri_z = uv[F], z[F]                        # (M,3,2), (M,3): triangle's corners
    keep = (tri_z > 0.1).all(axis=1)                   # remove triangles behind the camera
    tris = np.round(tri_uv[keep] * (1 << shift)).astype(np.int32)   # fixed-point sub-pixel coords
    mask = np.zeros(shape, np.uint8)
    for tri in tris:                                   # fillPoly on all at once XORs overlaps -> holes
        cv2.fillConvexPoly(mask, tri, 255, lineType=cv2.LINE_8, shift=shift)
    return mask

def outline(mask, min_pts=30):
    """External contours of a 0/255 mask, with at least min_pts points"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c for c in contours if len(c) >= min_pts]

def draw_overlay(img_bgr, mask, colour=(0, 255, 0)):
    out = img_bgr.copy()
    cv2.drawContours(out, outline(mask), -1, colour, 1)
    return out

def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / (a | b).sum()


def edge_map(img_bgr, clahe_clip=None, clahe_tile=(8, 8), blur_sigma=1.2, canny_low=10, canny_high=30, min_px=15):
    """Binary edge map of the observed image"""
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if clahe_clip is not None:
        grey = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile).apply(grey)
    grey = cv2.GaussianBlur(grey, (0, 0), sigmaX=blur_sigma)
    
    edges = cv2.Canny(grey, canny_low, canny_high, L2gradient=True)
    
    if min_px > 0:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= min_px
        keep[0] = False # background
        edges = np.where(keep[labels], 255, 0).astype(np.uint8)
    return edges

def dist_map(edges, tau=None):
    """Distance (in pixels) from every pixel to nearest edge pixel, optionally truncated at tau"""
    dt = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return dt if tau is None else np.minimum(dt, tau)

def oracle_edges(gt_mask):
    """Outline observed from the ground-truth mask"""
    edges = np.zeros_like(gt_mask, np.uint8)
    cv2.drawContours(edges, outline(gt_mask.astype(np.uint8) * 255), -1, 255, 1)
    return edges

def recall(edges, gt_mask, tol=2.0):
    """The fraction of the true lander outline that are with an edge found within the tolerace """
    pts = np.vstack(outline(gt_mask.astype(np.uint8) * 255)).reshape(-1, 2)
    pts = drop_border(pts, gt_mask.shape)
    d = dist_map(edges)
    return (d[pts[:, 1], pts[:, 0]] <= tol).mean()

def clutter(edges, gt_mask, near=4.0, far=50.0):
    """Edge pixels that are not on the true lander outline, that can help the optimiser"""
    d_obj = cv2.distanceTransform((gt_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return int(((edges > 0) & (d_obj >= near) & (d_obj <= far)).sum())

def drop_border(pts, shape, gap = 2):
    """Removes points lying on the image border (non real edges)"""
    h, w = shape[:2]
    x, y = pts[:, 0], pts[:, 1]
    keep = (x >= gap) & (x < w - gap) & (y >= gap) & (y < h - gap)
    return pts[keep]

def perturb(R0, C0, p):
    """Pose = base pose nudged by p = [tilt, pan, roll (deg), right, down, forward (m)], camera-frame."""
    dR = cv2.Rodrigues(np.radians(np.asarray(p[:3], float)))[0]
    R = dR @ R0                                    # rotate about the camera's own axes
    C = C0 + R0.T @ np.asarray(p[3:], float)       # camera-frame offset -> world
    return R, -R @ C

def contour_samples(mask, n=300, min_pts=30):
    """n evenly spaced outline points (x, y), excluding image-border truncation."""
    cs = outline(mask)
    if not cs:
        return None
    pts = drop_border(np.vstack(cs).reshape(-1, 2), mask.shape)
    if len(pts) < min_pts:
        return None
    return pts[np.linspace(0, len(pts), n, endpoint=False).astype(int)]


def make_ctx(V, F, K, R0, C0, obs_edges, tau=20.0, roi_pad=0.3):
    """Requirements for the cost function at a given pose and image."""
    m0 = project_mask(V, F, R0, -R0 @ C0, K, obs_edges.shape)
    x, y, w, h = cv2.boundingRect(m0)
    x0, y0 = max(x - int(w * roi_pad), 0), max(y - int(h * roi_pad), 0)
    x1, y1 = x + w + int(w * roi_pad), y + h + int(h * roi_pad)
    ys, xs = np.nonzero(obs_edges[y0:y1, x0:x1])          # observed edges in the FIXED ROI
    return dict(V=V, F=F, K=K, R0=R0, C0=C0, L=V.mean(axis=0), tau=tau, shape=obs_edges.shape,
                dt_obs=dist_map(obs_edges, tau),
                roi_pts=np.column_stack([xs + x0, ys + y0]))


def cost_terms(p, ctx):
    """(forward, reverse) truncated chamfer at pose p. Both in pixels, max tau."""
    R, t = perturb(ctx["R0"], ctx["C0"], p)
    mask = project_mask(ctx["V"], ctx["F"], R, t, ctx["K"], ctx["shape"])
    pts = contour_samples(mask)
    if pts is None:
        return ctx["tau"], ctx["tau"]
    forward = ctx["dt_obs"][pts[:, 1], pts[:, 0]].mean()       # model outline -> nearest observed edge

    model_edges = np.zeros(ctx["shape"], np.uint8)
    cv2.drawContours(model_edges, outline(mask), -1, 255, 1)
    rp = ctx["roi_pts"]
    reverse = dist_map(model_edges, ctx["tau"])[rp[:, 1], rp[:, 0]].mean() if len(rp) else 0.0
    return forward, reverse                                     # cost = forward + lam * reverse


def slices(ctx, rot_range=10, pos_range=3, steps=21):
    """Sweep each parameter alone around the base pose"""
    out = {}
    for i in range(6):
        span = rot_range if i < 3 else pos_range
        xs = np.linspace(-span, span, steps)
        terms = []
        for d in xs:
            p = np.zeros(6)
            p[i] = d
            terms.append(cost_terms(p, ctx))
        out[i] = (xs, np.array(terms))
    return out


def plot_slices(res, title, path, lam=0.5):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        xs, fr = res[i]
        ax.plot(xs, fr[:, 0], label="forward")
        ax.plot(xs, fr[:, 0] + lam * fr[:, 1], label=f"forward + {lam}·reverse")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel(SLICE_NAMES[i])
        ax.set_ylabel("cost (px)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)

def perturb_obj(R0, C0, L, p):
    """Object-centred nudge: rotate about the lander centre L, shift it in the camera frame.
    p = [rot x, y, z (deg), shift right, down, depth (m)]. Decouples image shift from shape change."""
    dR = cv2.Rodrigues(np.radians(np.asarray(p[:3], float)))[0]
    R = dR @ R0
    Lc = R0 @ (L - C0)                                   # lander centre in the base camera frame
    t = Lc + np.asarray(p[3:], float) - R @ L
    return R, -R.T @ t                                   # (R, C)


def cost(p, ctx, lam=0.0, trim=1.0):
    """Scalar cost = forward + lam * reverse"""
    R, C = perturb_obj(ctx["R0"], ctx["C0"], ctx["L"], p)
    mask = project_mask(ctx["V"], ctx["F"], R, -R @ C, ctx["K"], ctx["shape"])
    pts = contour_samples(mask)
    if pts is None:
        return ctx["tau"] * (1 + lam)
    d = ctx["dt_obs"][pts[:, 1], pts[:, 0]]
    if trim < 1.0:
        d = np.sort(d)[: max(1, int(trim * len(d)))]  # trim the worst outliers
    c = d.mean()
    if lam > 0 and len(ctx["roi_pts"]):
        model_edges = np.zeros(ctx["shape"], np.uint8)
        cv2.drawContours(model_edges, outline(mask), -1, 255, 1)
        rp = ctx["roi_pts"]
        c += lam * dist_map(model_edges, ctx["tau"])[rp[:, 1], rp[:, 0]].mean()
    return float(c)


# (tau px, rotation step deg, lateral step m, depth step m, trim fraction)
LEVELS = ((60.0, 5.0, 0.3, 1.0, 1.0), (20.0, 1.5, 0.08, 0.3, 1.0), (10.0, 0.5, 0.03, 0.1, 0.8))


def refine(V, F, K, R0, C0, obs_edges, lam=0.0, levels=LEVELS, restarts=2, maxfev=300):
    """Coarse-to-fine Nelder-Mead from (R0, C0). Returns R, C, final cost, evaluations used."""
    R, C, nfev, fun = R0, C0, 0, None
    for tau, rot, lat, dep, trim in levels:
        for _ in range(restarts):                       # fresh simplex beats NM's early stopping
            ctx = make_ctx(V, F, K, R, C, obs_edges, tau)
            simplex = np.vstack([np.zeros(6), np.diag([rot, rot, rot, lat, lat, dep])])
            res = minimize(cost, np.zeros(6), args=(ctx, lam, trim), method="Nelder-Mead",
                           options=dict(initial_simplex=simplex, xatol=0.01, fatol=0.01, maxfev=maxfev))
            R, C = perturb_obj(R, C, ctx["L"], res.x)
            nfev, fun = nfev + res.nfev, res.fun
    return R, C, fun, nfev


def add_noise(R, C, rot_deg, pos_m, rng):
    """Random rotation of exactly rot_deg and random camera offset of exactly pos_m."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    return cv2.Rodrigues(axis * np.radians(rot_deg))[0] @ R, C + d * pos_m


def pose_err(R, C, gt):
    """Same metric as PnP (single source of truth)."""
    pose = {"rvec": cv2.Rodrigues(R)[0], "tvec": (-R @ C).reshape(3, 1)}
    pos, rot, _ = pe.pose_error(pose, gt)
    return pos, rot

# ------------------------------------------------------------------ Global initialisation
SEARCH_SCALES = (0.75, 0.87, 1.0, 1.15, 1.3)    # covers 7–23 m from rings at 8/15/22 m
SEARCH_DOWNSAMPLE = 2                            # search at half resolution 
SEARCH_PAD = 150                                 # px of tau-padding so border-cut silhouettes can match
SEARCH_BAND = (2, 8)        # outer band, in downsampled px (≈ 4–16 px at full resolution)
SEARCH_BAND_W = 60.0 

def build_ref_db(V, F, K, ref_poses, shape=(960, 1280)):
    """Project the mesh at every reference pose once: outline points relative to the silhouette centroid."""
    db = []
    for frame, ref in sorted(ref_poses.items()):
        R, C = ref["R_world2cv"], ref["C"]
        mask = project_mask(V, F, R, -R @ C, K, shape)
        cs = outline(mask)
        if not cs:
            continue
        m = cv2.moments(mask, binaryImage=True)
        centroid = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
        db.append(dict(frame=frame, R=R, C=C, centroid=centroid, contours=[c.reshape(-1, 2) -centroid for c in cs]))
    return db


def _templates(contours, scale, ds, band=SEARCH_BAND):
    """Outline template + outer-band template on the same canvas, at scale/ds.
    Returns (line, ring, centroid position in canvas, n line px, n ring px)."""
    cs = [np.round(c * scale / ds).astype(np.int32) for c in contours]
    margin = band[1] + 2
    origin = np.vstack(cs).min(axis=0) - margin
    size = np.vstack(cs).max(axis=0) - origin + margin + 1
    cs = [(c - origin).reshape(-1, 1, 2) for c in cs]
    fill = np.zeros((size[1], size[0]), np.uint8)
    cv2.drawContours(fill, cs, -1, 255, -1)
    line = np.zeros_like(fill)
    cv2.drawContours(line, cs, -1, 1, 1)
    disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ring = (cv2.dilate(fill, disk(band[1])) > 0) & ~(cv2.dilate(fill, disk(band[0])) > 0)
    return line.astype(np.float32), ring.astype(np.float32), -origin, line.sum(), ring.sum()


def search_refs(edges, ref_db, tau=20.0, w=SEARCH_BAND_W, scales=SEARCH_SCALES,
                ds=SEARCH_DOWNSAMPLE, pad=SEARCH_PAD):
    """Global template search. Score = mean chamfer + w * edge density in a band just OUTSIDE
    the outline (true outline -> sparse background; shrunken outline inside the lander -> busy)."""
    dt = cv2.resize(dist_map(edges, tau), None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
    ed = cv2.resize((edges > 0).astype(np.float32), None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
    b = pad // ds
    dt = cv2.copyMakeBorder(dt, b, b, b, b, cv2.BORDER_CONSTANT, value=tau)
    ed = cv2.copyMakeBorder(ed, b, b, b, b, cv2.BORDER_CONSTANT, value=0)
    out = []
    for ref in ref_db:
        best = None
        for s in scales:
            line, ring, c_in, n, nr = _templates(ref["contours"], s, ds)
            if line.shape[0] >= dt.shape[0] or line.shape[1] >= dt.shape[1]:
                continue
            score = cv2.matchTemplate(dt, line, cv2.TM_CCORR) / n
            if w > 0:
                score += w * cv2.matchTemplate(ed, ring, cv2.TM_CCORR) / nr
            v, _, loc, _ = cv2.minMaxLoc(score)
            if best is None or v < best["score"]:
                best = dict(ref=ref, scale=s, score=v, centroid=(np.array(loc) + c_in) * ds - pad)
        if best:
            out.append(best)
    return sorted(out, key=lambda c: c["score"])


def _rotate_ray(K, px_from, px_to):
    """Camera rotation that turns the viewing ray of pixel px_from onto the ray of px_to."""
    Kinv = np.linalg.inv(K)
    a = Kinv @ np.array([*px_from, 1.0])
    b = Kinv @ np.array([*px_to, 1.0])
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    return np.eye(3) if s < 1e-12 else cv2.Rodrigues(axis / s * np.arctan2(s, a @ b))[0]


def start_pose(cand, K, L):
    """Reference pose -> start pose: range scaled by 1/scale, camera turned onto the found centroid."""
    ref, s = cand["ref"], cand["scale"]
    C = L + (ref["C"] - L) / s                      # bigger silhouette (s > 1) -> closer camera
    R = _rotate_ray(K, ref["centroid"], cand["centroid"]) @ ref["R"]
    return R, C


def localise_silhouette(V, F, K, ref_db, obs_edges, k=3, lam=0.0, levels=LEVELS):
    """Full silhouette localisation: global search -> k starts -> coarse level each -> refine the best."""
    L = V.mean(axis=0)
    cands = search_refs(obs_edges, ref_db, tau=levels[1][0])[:k]
    coarse = []
    for cand in cands:
        R0, C0 = start_pose(cand, K, L)
        R, C, fun, n = refine(V, F, K, R0, C0, obs_edges, lam, levels=levels[:1])
        coarse.append(dict(cost=fun, R=R, C=C, cand=cand, start=(R0, C0), nfev=n))
    best = min(coarse, key=lambda c: c["cost"])
    R, C, fun, n = refine(V, F, K, best["R"], best["C"], obs_edges, lam, levels=levels[1:])
    return R, C, fun, dict(ref_frame=best["cand"]["ref"]["frame"], scale=best["cand"]["scale"],
                           start=best["start"], nfev=sum(c["nfev"] for c in coarse) + n)

# if __name__ == "__main__":
#     mesh = SIL_MESH_PATH if SIL_MESH_PATH.exists() else MESH_PATH
#     V, F = load_mesh(mesh)
#     K = pe.load_intrinsics(pe.INTRINSICS_PATH)
#     poses = pe.load_poses(pe.TEST_DIR / "posev_testing.csv") 
#     RESULTS_DIR.mkdir(exist_ok=True)
#     out_csv = RESULTS_DIR / f"basin_{pe.SCENE}_{time.strftime('%Y-%m-%d_%H-%M')}.csv"


#     FRAMES = sorted(poses, key=lambda f: poses[f]["range"])[::7]
#     NOISE = [(2, 0.5), (5, 1.0), (10, 2.0)]
#     LAMS = [0.0]                                        # add 0.5 once lam=0 works
#     rng = np.random.default_rng(0)
#     rows = []

#     for frame in FRAMES:                
#         gt = poses[frame]
#         img = cv2.imread(str(pe.TEST_DIR / "RGB" / f"frame_{frame:04d}.png"))
#         _, index_map = pe.decompose_EXR(pe.TEST_DIR / "EXR" / f"frame_{frame:04d}.exr")
#         obs = {"oracle": oracle_edges(index_map == pe.LANDER_INDEX),
#             "edges": edge_map(img)
#         }       

#         for rot_deg, pos_m in NOISE:
#                     R0, C0 = add_noise(gt["R_world2cv"], gt["C"], rot_deg, pos_m, rng)   # same start for every variant
#                     for obs_name, obs_edges in obs.items():
#                         for lam in LAMS:
#                             t0 = time.perf_counter()
#                             R, C, fun, nfev = refine(V, F, K, R0, C0, obs_edges, lam)
#                             pos, rot = pose_err(R, C, gt)
#                             ok = pos < pe.POS_MAX_ERR and rot < pe.ROT_MAX_ERR
#                             rows.append(dict(frame=frame, range=gt["range"], noise_rot=rot_deg, noise_pos=pos_m,
#                                             obs=obs_name, lam=lam, pos_err=pos, rot_err=rot, ok=ok,
#                                             cost=fun, nfev=nfev, runtime=time.perf_counter() - t0))
#                             print(f"f{frame:3d} {rot_deg:>2}°/{pos_m}m {obs_name:6s} lam {lam}: "
#                                 f"pos {pos:.3f} m rot {rot:.2f}° {'OK' if ok else '--'}  "
#                                 f"cost {fun:.2f} nfev {nfev} {rows[-1]['runtime']:.0f}s")
#                             pd.DataFrame(rows).to_csv(out_csv, index=False)   # save as you go

#     df = pd.DataFrame(rows)
#     print(df.groupby(["obs", "lam", "noise_rot"])[["ok", "pos_err", "rot_err", "runtime"]]
#             .agg({"ok": "mean", "pos_err": "median", "rot_err": "median", "runtime": "median"}).round(3))