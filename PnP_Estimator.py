import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import torch
import glob
import OpenEXR

def decomposeEXR(exr_path):
    # Load the exr file
    with OpenEXR.File(exr_path) as exr_file:
        part = exr_file.parts[0]
        channels = part.channels

        print("Available channels inside EXR:", list(channels.keys()))

        pos_x = channels.get("ViewLayer.Position.R")
        pos_y = channels.get("ViewLayer.Position.G")
        pos_z = channels.get("ViewLayer.Position.B")

        if pos_x and pos_y and pos_z:
            position_map = np.stack([pos_x.pixels, pos_y.pixels, pos_z.pixels], axis=-1)
        else:
            print("Warning: Position pass missing. Check Blender render pass settings.")
            position_map = None

        # extract object index pass
        obj_index = channels.get("ViewLayer.IndexOB")

        if obj_index:
            object_index_map = obj_index.pixels.astype(np.uint32)
        else:
            print("Warning: Object Index pass missing.")
            object_index_map = None

    return position_map, object_index_map

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

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ORB feature detection
    keypoints_orb, descriptors_orb = orb.detectAndCompute(gray_image, None)

    # AKAZE feature detection
    keypoints_akaze, descriptors_akaze = akaze.detectAndCompute(gray_image, None)

    # LightGlue feature detection
    inputs = processor(images=image_rgb, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    #position map and object index map extraction



