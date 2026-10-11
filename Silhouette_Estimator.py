import PnP_Estimator as pe
import csv
import argparse 
import OpenEXR
import json
import numpy as np
from pathlib import Path
import cv2
import time
import pandas



MESH_PATH = Path(__file__).resolve().parent / "Project_Intrinsics" / f"LAND_{pe.SCENE}.obj"
RESULTS_DIR = Path(__file__).resolve().parent / "results_silhouette"


def load_mesh(path):
    V = []
    F = []  

    with open(path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    V.append([float(x) for x in line.split()[1:]])
                elif line.startswith("f "):
                    F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:]])

    V = np.array(V, dtype=np.float32)
    F = np.array(F, dtype=np.int32)
    return V, F

def project_points(V, R, t, K):
    """ Converts 3D world points to 2D pixel coods and camera depth"""
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
    """"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c for c in contours if len(c) >= min_pts]

def draw_overlay(img_bgr, mask, colour=(0, 255, 0)):
    out = img_bgr.copy()
    cv2.drawContours(out, outline(mask), -1, colour, 1)
    return out

def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / (a | b).sum()


if __name__ == "__main__":
    V,F = load_mesh(MESH_PATH)
    K = pe.load_intrinsics(pe.INTRINSICS_PATH)
    poses = pe.load_poses(pe.TEST_DIR / "posev_testing.csv")
    out_dir = pe.DEBUG_DIR / "silhouette"
    out_dir.mkdir(parents=True, exist_ok=True)


    for frame in (99, 34, 15, 31, 57):                 # near, near, mid, mid, far
        gt = poses[frame]
        R = gt["R_world2cv"]
        t = -R @ gt["C"]

        t0 = time.perf_counter()
        mask = project_mask(V, F, R, t, K)
        dt = time.perf_counter() - t0

        _, index_map = pe.decompose_EXR(pe.TEST_DIR / "EXR" / f"frame_{frame:04d}.exr")
        gt_mask = index_map == pe.LANDER_INDEX        
        img = cv2.imread(str(pe.TEST_DIR / "RGB" / f"frame_{frame:04d}.png"))
        cv2.imwrite(str(out_dir / f"{pe.SCENE}_frame_{frame:04d}.png"), draw_overlay(img, mask))

        print(f"frame {frame:3d}  range {gt['range']:5.1f} m  IoU {iou(mask, gt_mask):.4f}  project {dt*1000:.0f} ms")
