import PnP_Estimator as pe
import numpy as np
from pathlib import Path
import cv2
import time



MESH_PATH = Path(__file__).resolve().parent / "Project_Intrinsics" / f"LAND_{pe.SCENE}.obj"
RESULTS_DIR = Path(__file__).resolve().parent / "results_silhouette"

CONFIGS = {
    "report (CLAHE, 50/150)": dict(clahe_clip=2.0, blur_sigma=1.0, canny_low=50, canny_high=150),
    "no CLAHE, s1.2, 10/30":  dict(clahe_clip=None, blur_sigma=1.2, canny_low=10, canny_high=30),
    "CLAHE, s1.2, 10/30":     dict(clahe_clip=2.0, blur_sigma=1.2, canny_low=10, canny_high=30),
    "no CLAHE, s2.0, 20/50":  dict(clahe_clip=None, blur_sigma=2.0, canny_low=20, canny_high=50),
}


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

if __name__ == "__main__":
    out_dir = pe.DEBUG_DIR / "silhouette"
    out_dir.mkdir(parents=True, exist_ok=True)


    for frame in (99, 34, 15, 31, 57):                 # near, near, mid, mid, far

        img = cv2.imread(str(pe.TEST_DIR / "RGB" / f"frame_{frame:04d}.png"))
        _, index_map = pe.decompose_EXR(pe.TEST_DIR / "EXR" / f"frame_{frame:04d}.exr")
        gt_mask = index_map == pe.LANDER_INDEX        

        for i in CONFIGS:
            print(f"frame {frame:3d}  config {i}")
            edges = edge_map(img, **CONFIGS[i])
            dt = dist_map(edges, tau=20)

            cv2.imwrite(str(out_dir / f"{pe.SCENE}_edges_{frame:04d}_{i}.png"), edges)
            cv2.imwrite(str(out_dir / f"{pe.SCENE}_dist_{frame:04d}_{i}.png"), (dt / 20 * 255).astype(np.uint8))

            print(f"frame {frame:3d}  recall {recall(edges, gt_mask):.3f} clutter {clutter(edges, gt_mask):6d}" 
                  f" oracle recall {recall(oracle_edges(gt_mask), gt_mask):.3f}")