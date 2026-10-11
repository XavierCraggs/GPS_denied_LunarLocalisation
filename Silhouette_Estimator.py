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


if __name__ == "__main__":
    V,F = load_mesh(MESH_PATH)
    print(V.shape, F.shape)
    print(V.min(0), V.max(0))
