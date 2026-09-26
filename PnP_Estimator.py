import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import torch
import glob
import OpenEXR
import json
import re
from pathlib import Path

##setup
ROOT = Path("/Users/xaviercraggs/Library/CloudStorage/OneDrive-Personal/QUT/Thesis/render_output")
REF_DIR = ROOT / "TRAINING"
TEST_DIR = ROOT / "TESTING"
INTRINSICS_PATH = Path(__file__).parent / "Project_Intrinsics" / "Camera_details.json"

LANDER_INDEX = 1 
LIGHTGLUE_THRESHOLD = 0.2
LIGHTGLUE_BATCH = 8

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
        channels = exr_file.parts[0].channels

        position_map = channels["Position"].pixels[..., :3].astype(np.float64) 
        object_index_map = channels["IndexOB.V"].pixels.astype(np.int32)

    return position_map, object_index_map

def frame_number(path):
    return int(re.findall(r"\d+", Path(path).stem)[-1])

def build_reference_db(ref_dir, orb, akaze):
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

        reference_db.append({
            "frame": frame,
            "image_rgb": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "lander_mask": lander_mask,
            "position_map": position_map,
            "object_index_map": object_index_map,
            "orb": orb.detectAndCompute(gray, lander_mask),
            "akaze": akaze.detectAndCompute(gray, lander_mask),
        })

    print(f"Reference DB: {len(reference_db)} frames")
    return reference_db

def bilinear_interpolate(image, x, y):
    h, w = image.shape[:2]

    # Ensure x and y are within the image bounds
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    x0 = int(np.floor(x))
    x1 = min(x0 + 1, w - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, h - 1)

    dx = x - x0
    dy = y - y0

    top_left = image[y0, x0]
    top_right = image[y0, x1]
    bottom_left = image[y1, x0]
    bottom_right = image[y1, x1]

    top = (1 - dx) * top_left + dx * top_right
    bottom = (1 - dx) * bottom_left + dx * bottom_right

    return (1 - dy) * top + dy * bottom



def match_lightglue(test_rgb, reference_db, processor, model):
    h, w = test_rgb.shape[:2]
    scale = torch.tensor([w, h], device=model.device, dtype=torch.float32)
    all_matches = []

    for start in range(0, len(reference_db), LIGHTGLUE_BATCH):
        chunk = reference_db[start:start + LIGHTGLUE_BATCH]
        pairs = [[test_rgb, ref["image_rgb"]] for ref in chunk]

        inputs = processor(images=pairs, return_tensors="pt", do_resize=False).to(model.device)
        with torch.inference_mode():
            out = model(**inputs)

        keypoints = out.keypoints.float() * scale   # (B, 2, N, 2) relative -> pixels

        for i, ref in enumerate(chunk):
            mask0 = out.mask[i, 0] > 0
            mask1 = out.mask[i, 1] > 0
            kp_test = keypoints[i, 0][mask0]
            kp_ref = keypoints[i, 1][mask1]
            matches = out.matches[i, 0][mask0]
            scores = out.matching_scores[i, 0][mask0]

            valid = (scores > LIGHTGLUE_THRESHOLD) & (matches > -1) & (matches < kp_ref.shape[0])
            pts_test = kp_test[valid].cpu().numpy().astype(np.float64)
            pts_ref = kp_ref[matches[valid]].cpu().numpy().astype(np.float64)
            match_scores = scores[valid].cpu().numpy()

            # Only keep matches on the lander
            u = np.clip(np.rint(pts_ref[:, 0]).astype(int), 0, w - 1)
            v = np.clip(np.rint(pts_ref[:, 1]).astype(int), 0, h - 1)
            on_lander = ref["lander_mask"][v, u] > 0

            all_matches.append({
                "ref_frame": ref["frame"],
                "pts_test": pts_test[on_lander],   # (M, 2) pixels in test image
                "pts_ref": pts_ref[on_lander],     # (M, 2) pixels in reference image
                "scores": match_scores[on_lander],
            })

    return all_matches


# =======
#  Main
# =======
def main():
    orb = cv2.ORB_create(nfeatures=3000)
    akaze = cv2.AKAZE_create()

    processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint").to(device)
    model.eval()

    reference_db = build_reference_db(REF_DIR, orb, akaze)
    if not reference_db:
        raise RuntimeError(f"No reference frames found in {REF_DIR}")

    K = load_intrinsics(INTRINSICS_PATH)
    print("K =\n", K)

    for test_path in sorted((TEST_DIR / "RGB").glob("*.png")):
        image = cv2.imread(str(test_path))
        if image is None:
            print(f"Failed to load test image: {test_path}")
            continue

        # Test RGB 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        kp_orb, des_orb = orb.detectAndCompute(gray, None)
        kp_akaze, des_akaze = akaze.detectAndCompute(gray, None)
        lightglue_matches = match_lightglue(image_rgb, reference_db, processor, model)

        best = max(lightglue_matches, key=lambda m: len(m["pts_test"]))
        print(f"{test_path.name}: ORB {len(kp_orb)} kp, AKAZE {len(kp_akaze)} kp, "
              f"Best LightGlue {best['ref_frame']} with {len(best['pts_test'])} matches")


if __name__ == "__main__":
    main()