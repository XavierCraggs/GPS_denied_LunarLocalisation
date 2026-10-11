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

R_BCAM2CV = np.diag([1.0, -1.0, -1.0])   # Blender camera axes -> OpenCV camera axes

POS_MAX_ERR = 0.5   # Maximuim position and rotation error
ROT_MAX_ERR = 5.0

RESULTS_DIR = Path(__file__).resolve().parent / "results_silhouette"

pe.load_intrinsics
pe.decompose_EXR
pe.load_poses
pe.quat_to_matrix
pe.pose_error
pe.summarise
pe.save_results
pe.load_poses
pe.parse_args