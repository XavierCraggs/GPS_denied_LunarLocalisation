import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import torch
import glob

def decomposeEXR(exr_path):
    # Load the exr file
    exr = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)

    # Extract the position layer


#Exr = np.float32([0,0,0])

exr = cv2.imread("exr_file.exr", cv2.IMREAD_UNCHANGED) #get actual file path

image_paths = glob.glob("path_to_images/*.png") #get actual file path

#initialise three detectors
orb = cv2.ORB_create()

akaze = cv2.AKAZE_create()

processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint", device_map="auto")


#main loop to process images
for path in image_paths:
    image = cv2.imread(path) #get actual file path

    if image is None:
        print(f"Failed to load image: {path}")
        continue

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ORB feature detection
    keypoints_orb, descriptors_orb = orb.detectAndCompute(gray_image, None)

    # AKAZE feature detection
    keypoints_akaze, descriptors_akaze = akaze.detectAndCompute(gray_image, None)

    # LightGlue feature detection
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)


