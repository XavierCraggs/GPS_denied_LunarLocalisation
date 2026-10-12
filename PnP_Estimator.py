import cv2
import numpy as np
import OpenEXR
import json
import re
from pathlib import Path
import argparse
import shutil
import csv
import time
import pandas as pd

##setup
SCENE = "v1"

ROOT = Path(__file__).resolve().parent.parent / f"render_output_{SCENE}"
REF_DIR = ROOT / "TRAINING"
TEST_DIR = ROOT / "TESTING"
DEBUG_DIR = Path(__file__).resolve().parent / "debug"
INTRINSICS_PATH = Path(__file__).parent / "Project_Intrinsics" / "Camera_details.json"

LANDER_INDEX = 1 
LIGHTGLUE_THRESHOLD = 0.2
LIGHTGLUE_BATCH = 4

RATIO_TEST = 0.8   # Lowe's ratio test threshold

EDGE_JUMP_M = 0.05          # max 3D spread (m) within a 2x2 pixel cell before it's treated as an edge
PNP_REPROJ_PX = 3.0         # RANSAC inlier threshold (pixels)
PNP_ITERATIONS = 1000
PNP_CONFIDENCE = 0.999
MIN_CORRESPONDENCES = 6     # minimum 2D-3D pairs to attempt PnP
MIN_INLIERS = 10            # minimum RANSAC inliers to accept a pose

POS_MAX_ERR = 0.5   # Maximuim position and rotation error
ROT_MAX_ERR = 5.0

R_BCAM2CV = np.diag([1.0, -1.0, -1.0])   # Blender camera axes -> OpenCV camera axes
RESULTS_DIR = Path(__file__).resolve().parent / "results"

def save_debug_images(test_bgr, test_name, ref, match, method, max_lines=80):
    out_dir = DEBUG_DIR / test_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reference lander mask (should be a white lander on black)
    cv2.imwrite(str(out_dir / f"mask_ref{ref['frame']:04d}.png"), ref["lander_mask"])

    

    # Side-by-side: test (left) | reference (right)
    ref_bgr = cv2.cvtColor(ref["image_rgb"], cv2.COLOR_RGB2BGR)
    canvas = np.hstack([test_bgr, ref_bgr])
    offset = test_bgr.shape[1]

    # Draw only the strongest matches so the image stays readable
    order = np.argsort(-match["scores"])[:max_lines]
    for i in order:
        p1 = tuple(np.round(match["pts_test"][i]).astype(int))
        p2 = tuple(np.round(match["pts_ref"][i] + [offset, 0]).astype(int))
        cv2.line(canvas, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, (0, 0, 255), -1)
        cv2.circle(canvas, p2, 3, (0, 0, 255), -1)

    cv2.imwrite(str(out_dir / f"matches_{method}_ref{ref['frame']:04d}.png"), canvas)
    print(f"  debug images -> {out_dir}")

def load_intrinsics(json_path):
    with open(json_path) as f:
        cam = json.load(f)

    return np.array([
        [cam["fx"], 0.0,       cam["cx"]],
        [0.0,       cam["fy"], cam["cy"]],
        [0.0,       0.0,       1.0],
    ], dtype=np.float64)

def decompose_EXR(exr_path):
    # Load the exr file
    with OpenEXR.File(str(exr_path)) as exr_file:
        channels = {}
        for part in exr_file.parts:
            channels.update(part.channels)

        position_map = channels["Position"].pixels[..., :3].astype(np.float64) 
        object_index_map = channels["IndexOB.V"].pixels.astype(np.int32)

    return position_map, object_index_map

def frame_number(path):
    return int(re.findall(r"\d+", Path(path).stem)[-1])

def build_reference_db(ref_dir, orb, akaze, processor, model):
    rgb_paths = {frame_number(p): p for p in (ref_dir / "RGB").glob("*.png")}
    exr_paths = {frame_number(p): p for p in (ref_dir / "EXR").glob("*.exr")}

    unpaired = rgb_paths.keys() ^ exr_paths.keys()
    if unpaired:
        print(f"Skipping frames without both PNG and EXR: {sorted(unpaired)}")

    reference_db = []
    for frame in sorted(rgb_paths.keys() & exr_paths.keys()):
        image = cv2.imread(str(rgb_paths[frame]))
        if image is None:
            print(f"Failed to load reference image: {rgb_paths[frame]}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        position_map, object_index_map = decompose_EXR(exr_paths[frame])

        if position_map.shape[:2] != gray.shape:
            raise ValueError(f"Ref frame {frame}: EXR {position_map.shape[:2]} != PNG {gray.shape}")

        # Segment the lander
        lander_mask = (object_index_map == LANDER_INDEX).astype(np.uint8) * 255

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        reference_db.append({
            "frame": frame,
            "image_rgb": image_rgb,
            "lander_mask": lander_mask,
            "position_map": position_map,
            "object_index_map": object_index_map,
            "orb": orb.detectAndCompute(gray, lander_mask),
            "akaze": akaze.detectAndCompute(gray, lander_mask),
            "superpoint": extract_superpoint(image_rgb, processor, model),
        })

    print(f"Reference DB: {len(reference_db)} frames")
    return reference_db

def lift_to_3d(pts_ref, position_map, object_index_map):
    """Bilinearly sample world XYZ at sub-pixel reference keypoints.
    Rejects points whose 2x2 sampling cell touches the background, another
    object, or a depth discontinuity (silhouette / self-occlusion edge)."""
    h, w = object_index_map.shape
    u, v = pts_ref[:, 0], pts_ref[:, 1]

    x0 = np.floor(u).astype(int)
    y0 = np.floor(v).astype(int)
    inside = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    x0 = np.clip(x0, 0, w - 2)
    y0 = np.clip(y0, 0, h - 2)
    x1, y1 = x0 + 1, y0 + 1

    # The four neighbouring pixels used by bilinear interpolation
    corners = [(y0, x0), (y0, x1), (y1, x0), (y1, x1)]
    P = np.stack([position_map[yy, xx] for yy, xx in corners], axis=1)       # (N, 4, 3)
    on_lander = np.all(np.stack([object_index_map[yy, xx] == LANDER_INDEX
                                 for yy, xx in corners], axis=1), axis=1)    # (N,)

    # Depth-discontinuity check: all 4 corners must lie close together in 3D
    spread = np.linalg.norm(P - P.mean(axis=1, keepdims=True), axis=2).max(axis=1)
    valid = inside & on_lander & (spread < EDGE_JUMP_M)

    dx = (u - x0)[:, None]
    dy = (v - y0)[:, None]
    top = (1 - dx) * P[:, 0] + dx * P[:, 1]
    bottom = (1 - dx) * P[:, 2] + dx * P[:, 3]
    pts_3d = (1 - dy) * top + dy * bottom

    return pts_3d, valid


def estimate_pose(pts_3d, pts_2d, K):
    """RANSAC PnP, then Levenberg-Marquardt refinement on the inliers."""
    if len(pts_3d) < MIN_CORRESPONDENCES:
        return None
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, K, None,
            iterationsCount=PNP_ITERATIONS,
            reprojectionError=PNP_REPROJ_PX,
            confidence=PNP_CONFIDENCE,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    if not ok or inliers is None or len(inliers) < MIN_INLIERS:
        return None

    idx = inliers.ravel()
    try:
        rvec, tvec = cv2.solvePnPRefineLM(pts_3d[idx], pts_2d[idx], K, None, rvec, tvec)
    except cv2.error:
        pass
    proj, _ = cv2.projectPoints(pts_3d[idx], rvec, tvec, K, None)
    reproj_err = np.linalg.norm(proj.reshape(-1, 2) - pts_2d[idx], axis=1).mean()

    return {"rvec": rvec, "tvec": tvec, "inliers": len(idx), "reproj_px": reproj_err}


def localise(matches, reference_db, K):
    """Run PnP against every reference; keep the one with the most RANSAC inliers."""
    refs = {r["frame"]: r for r in reference_db}
    best = None

    for m in matches:
        if len(m["pts_test"]) < MIN_CORRESPONDENCES:
            continue
        ref = refs[m["ref_frame"]]
        pts_3d, valid = lift_to_3d(m["pts_ref"], ref["position_map"], ref["object_index_map"])
        pose = estimate_pose(pts_3d[valid], m["pts_test"][valid], K)
        if pose is None:
            continue
        pose.update(ref_frame=m["ref_frame"], n_matches=len(m["pts_test"]), n_lifted=int(valid.sum()))
        if best is None or pose["inliers"] > best["inliers"]:
            best = pose

    return best


def quat_to_matrix(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])


def load_poses(csv_path):
    """Ground truth per frame: camera centre (world) and world->OpenCV-camera rotation."""
    poses = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            R_bcam2world = quat_to_matrix(*(float(row[k]) for k in ("qw", "qx", "qy", "qz")))
            poses[int(row["frame"])] = {
                "C": np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
                "R_world2cv": R_BCAM2CV @ R_bcam2world.T,
                "range": float(row["range"]),
                "azimuth": float(row["azimuth"]),
            }
    return poses


def pose_error(pose, gt):
    R_est, _ = cv2.Rodrigues(pose["rvec"])
    C_est = (-R_est.T @ pose["tvec"]).ravel()

    pos_err = np.linalg.norm(C_est - gt["C"])
    cos_angle = (np.trace(R_est @ gt["R_world2cv"].T) - 1) / 2
    rot_err = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return pos_err, rot_err, C_est


def save_results(results):
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"pnp_results_{SCENE}_{time.strftime('%Y%m%d_%H%M')}.csv"
    fields = ["test_frame", "method", "success", "ref_frame", "gt_range", "gt_azimuth",
              "n_matches", "n_lifted", "n_inliers", "reproj_px",
              "pos_err_m", "rot_err_deg", "est_x", "est_y", "est_z", "runtime_s"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved -> {out_path}")

def summarise(df):
    df["pose"] = df["success"].astype(str).str.lower() == "true"
    df["ok"] = df["pose"] & (df["pos_err_m"] < POS_MAX_ERR) & (df["rot_err_deg"] < ROT_MAX_ERR)
    df["range"] = pd.cut(df["gt_range"], [0, 11.5, 18.5, 30], labels=["near", "mid", "far"])
    ok = df[df["ok"]]

    summary = pd.DataFrame({
        "success_%":   df.groupby("method")["ok"].mean() * 100,
        "failures":    df.groupby("method")["ok"].apply(lambda s: (~s).sum()),
        "pos_med_cm":  ok.groupby("method")["pos_err_m"].median() * 100,
        "pos_p90_cm":  ok.groupby("method")["pos_err_m"].quantile(0.9) * 100,
        "rot_med_deg": ok.groupby("method")["rot_err_deg"].median(),
        "runtime_s":   df.groupby("method")["runtime_s"].median(),
    }).round(2)
    by_range = (ok.groupby(["range", "method"], observed=True)["pos_err_m"].median() * 100).unstack().round(1)

    print(f"\nSuccess = pose found, pos < {POS_MAX_ERR} m, rot < {ROT_MAX_ERR} deg\n")
    print(summary, "\n")
    print("Median position error (cm) by range:\n", by_range, "\n")
    print("Failures:\n", df.loc[~df["ok"], ["test_frame", "method", "gt_azimuth", "pos_err_m", "rot_err_deg"]]
          .round(2).to_string(index=False))

def match_binary(kp_test, des_test, reference_db, method):
    """ORB + AKAZE: kNN + Lowe ratio test against each training frame."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    all_matches = []

    for ref in reference_db:
        kp_ref, des_ref = ref[method]
        good = []

        if des_test is not None and des_ref is not None and len(des_ref) >= 2:
            for pair in bf.knnMatch(des_test, des_ref, k=2):
                if len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance:
                    good.append(pair[0])

        all_matches.append({
            "ref_frame": ref["frame"],
            "pts_test": np.array([kp_test[m.queryIdx].pt for m in good], dtype=np.float64).reshape(-1, 2),
            "pts_ref":  np.array([kp_ref[m.trainIdx].pt for m in good], dtype=np.float64).reshape(-1, 2),
            "scores":   np.array([-m.distance for m in good], dtype=np.float64),  # higher = better
        })

    return all_matches

def extract_superpoint(image_rgb, processor, model):
    """Run SuperPoint once on one image. Returns pixel keypoints (N,2) and descriptors (N,256) on the model's device."""
    import torch
    
    h, w = image_rgb.shape[:2]
    # The processor only accepts image pairs, so pass the image twice and keep the first
    pixel_values = processor(images=[[image_rgb, image_rgb]], return_tensors="pt",
                             do_resize=False)["pixel_values"][:, 0].to(model.device)
    with torch.inference_mode():
        keypoints, _, descriptors, mask = model.keypoint_detector(pixel_values)[:4]
    valid = mask[0] > 0
    scale = torch.tensor([w, h], device=model.device, dtype=torch.float32)
    return {
        "keypoints": keypoints[0][valid].float() * scale,   # (N, 2) absolute pixels
        "descriptors": descriptors[0][valid].float(),       # (N, 256)
        "size": (h, w),
    }


def _pad(t, n):
    """Zero-pad the first dimension of t to length n."""
    import torch
    return torch.cat([t, t.new_zeros((n - t.shape[0],) + t.shape[1:])], dim=0)

def match_lightglue(test_feats, reference_db, model):
    """Match cached SuperPoint features of the test image against cached reference features."""
    import torch
    h, w = test_feats["size"]
    all_matches = []

    for start in range(0, len(reference_db), LIGHTGLUE_BATCH):
        chunk = reference_db[start:start + LIGHTGLUE_BATCH]
        n_max = max([test_feats["keypoints"].shape[0]] +
                    [r["superpoint"]["keypoints"].shape[0] for r in chunk])

        kps, descs, masks = [], [], []
        for ref in chunk:
            pair = (test_feats, ref["superpoint"])
            kps.append(torch.stack([_pad(f["keypoints"], n_max) for f in pair]))
            descs.append(torch.stack([_pad(f["descriptors"], n_max) for f in pair]))
            masks.append(torch.stack([_pad(torch.ones(f["keypoints"].shape[0], device=model.device), n_max)
                                      for f in pair]))

        with torch.inference_mode():
            matches, scores = model._match_image_pair(
                torch.stack(kps), torch.stack(descs), h, w, mask=torch.stack(masks))[:2]

        for i, ref in enumerate(chunk):
            n_test = test_feats["keypoints"].shape[0]
            n_ref = ref["superpoint"]["keypoints"].shape[0]
            m = matches[i, 0, :n_test]
            s = scores[i, 0, :n_test]
            valid = (s > LIGHTGLUE_THRESHOLD) & (m > -1) & (m < n_ref)

            pts_test = test_feats["keypoints"][valid].cpu().numpy().astype(np.float64)
            pts_ref = ref["superpoint"]["keypoints"][m[valid]].cpu().numpy().astype(np.float64)
            match_scores = s[valid].cpu().numpy()

            # Only keep matches on the lander
            u = np.clip(np.rint(pts_ref[:, 0]).astype(int), 0, w - 1)
            v = np.clip(np.rint(pts_ref[:, 1]).astype(int), 0, h - 1)
            on_lander = ref["lander_mask"][v, u] > 0

            all_matches.append({
                "ref_frame": ref["frame"],
                "pts_test": pts_test[on_lander],
                "pts_ref": pts_ref[on_lander],
                "scores": match_scores[on_lander],
            })

    return all_matches


def parse_args():
    parser = argparse.ArgumentParser(description="PnP localisation: ORB / AKAZE / LightGlue")
    parser.add_argument("--frames", type=int, nargs="+",
                        help="specific test frame numbers to run, e.g. --frames 0 3 7")
    parser.add_argument("--limit", type=int,
                        help="only run the first N test frames")
    parser.add_argument("--debug", action="store_true",
                        help="save lander mask and match visualisations")
    parser.add_argument("--summary", type=Path,
                         help="only summarise an existing results CSV")
    return parser.parse_args()


# =======
#  Main
# =======
def main():
    import torch
    from transformers import AutoImageProcessor, AutoModel

    args = parse_args()

    if args.summary:
        summarise(pd.read_csv(args.summary))
        return

    orb = cv2.ORB_create(nfeatures=3000)
    akaze = cv2.AKAZE_create()

    processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint").to(device)
    model.eval()

    reference_db = build_reference_db(REF_DIR, orb, akaze, processor, model)
    if not reference_db:
        raise RuntimeError(f"No reference frames found in {REF_DIR}")

    K = load_intrinsics(INTRINSICS_PATH)
    print("K =\n", K)

    test_paths = sorted((TEST_DIR / "RGB").glob("*.png"))
    if args.frames is not None:
        wanted = set(args.frames)
        test_paths = [p for p in test_paths if frame_number(p) in wanted]
    if args.limit is not None:
        test_paths = test_paths[:args.limit]

    if not test_paths:
        raise RuntimeError("No test frames selected")
    print(f"Running {len(test_paths)} test frame(s)")

    test_poses = load_poses(TEST_DIR / "posev_testing.csv")
    results = []

    match_lightglue(reference_db[0]["superpoint"], reference_db[:LIGHTGLUE_BATCH], model)

    for test_path in test_paths:
        image = cv2.imread(str(test_path))
        if image is None:
            print(f"Failed to load test image: {test_path}")
            continue

        frame = frame_number(test_path)
        gt = test_poses[frame]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        methods = {
            "orb":       lambda: match_binary(*orb.detectAndCompute(gray, None), reference_db, "orb"),
            "akaze":     lambda: match_binary(*akaze.detectAndCompute(gray, None), reference_db, "akaze"),
            "lightglue": lambda: match_lightglue(extract_superpoint(image_rgb, processor, model), reference_db, model),
        }

        if args.debug:
            shutil.rmtree(DEBUG_DIR / test_path.stem, ignore_errors=True)

        print(f"{test_path.name}  (GT range {gt['range']:.1f} m, azimuth {gt['azimuth']:.0f} deg)")
        for method, run_matching in methods.items():
            t0 = time.perf_counter()
            matches = run_matching()
            pose = localise(matches, reference_db, K)
            runtime = time.perf_counter() - t0

            row = {"test_frame": frame, "method": method,
                   "gt_range": gt["range"], "gt_azimuth": gt["azimuth"], "runtime_s": runtime}

            if pose is None:
                row.update(success=False)
                print(f"  {method:9s} FAILED (no reference gave >= {MIN_INLIERS} inliers)  {runtime:.2f}s")
            else:
                pos_err, rot_err, C_est = pose_error(pose, gt)
                row.update(success=True, ref_frame=pose["ref_frame"], n_matches=pose["n_matches"],
                           n_lifted=pose["n_lifted"], n_inliers=pose["inliers"],
                           reproj_px=pose["reproj_px"], pos_err_m=pos_err, rot_err_deg=rot_err,
                           est_x=C_est[0], est_y=C_est[1], est_z=C_est[2])
                print(f"  {method:9s} ref {pose['ref_frame']:3d}  inliers {pose['inliers']:4d}/{pose['n_lifted']:4d}  "
                      f"pos {pos_err:.3f} m  rot {rot_err:.2f} deg  reproj {pose['reproj_px']:.2f}px  {runtime:.2f}s")

                if args.debug:
                    match = next(m for m in matches if m["ref_frame"] == pose["ref_frame"])
                    best_ref = next(r for r in reference_db if r["frame"] == pose["ref_frame"])
                    save_debug_images(image, test_path.stem, best_ref, match, method)

            results.append(row)

    save_results(results)
    summarise(pd.DataFrame(results))


if __name__ == "__main__":
    main()