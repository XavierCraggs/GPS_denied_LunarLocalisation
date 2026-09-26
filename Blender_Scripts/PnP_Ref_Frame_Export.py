import bpy
import csv
import math
from mathutils import Vector
import os
import random
import shutil



## Constants
MODE = 'TESTING'  # Options: 'TRAINING' or 'TESTING'
NUM_TEST_FRAMES = 10  # Number of test frames to generate if in testing MODE
CAMERA_HEIGHT = 1.0  # Height of the camera above the terrain
Camera = bpy.data.objects['Rover_Camera']  
Terrain = bpy.data.objects['Moon_Terrain']  
Lander = bpy.data.objects['LAND_Base']

#Paths for saved training frames
DIR = bpy.path.abspath("//render_output/") 
MODE_DIR = os.path.join(DIR, MODE)
RGB = os.path.join(MODE_DIR, "RGB")
EXR = os.path.join(MODE_DIR, "EXR")

RESOLUTION_X = 1280
RESOLUTION_Y = 960



def cam_look_at(camera, target):
    track_to = camera.constraints.new(type='TRACK_TO')

    # Configure the constraint properties
    track_to.target = target
    track_to.track_axis = 'TRACK_NEGATIVE_Z'  # Blender cameras point down their local -Z axis
    track_to.up_axis = 'UP_Y'                 # Keeps the camera upright relative to its Y axis


#Start the camera tracking the lander (auto updates)
cam_look_at(Camera, Lander)

#tag the lander as '1' to easily identify in depth map
def tag_lander(obj):
    stack = [obj]

    while stack: 
        o = stack.pop()
        o.pass_index = 1
        stack.extend(o.children)

#GPU setup
def setup_gpu():
    scene = bpy.context.scene

    scene.render.engine = 'CYCLES'

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()

    for d in prefs.devices:
        d.use = d.type =="METAL"

    scene.cycles.device = "GPU"
    scene.render.use_persistent_data = False
    scene.cycles.samples = 64
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"

    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    layer = bpy.context.view_layer
    layer.use_pass_combined = True
    layer.use_pass_position = True
    layer.use_pass_object_index = True
    layer.use_pass_z = True

    node_tree = bpy.data.node_groups.new("Lunar_PnP_Compositor", "CompositorNodeTree")
    scene.compositing_node_group = node_tree    

    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    render_layers = nodes.new("CompositorNodeRLayers")

    # png = nodes.new("CompositorNodeOutputFile")
    # png.base_path = RGB
    # png.format.file_format = "PNG"
    # png.format.color_mode = "RGB" 
    # png.file_slots.new("frame_")

    # links.new(
    #     render_layers.outputs["Image"],
    #     png.inputs["frame_"]
    # )

    exr = nodes.new("CompositorNodeOutputFile")
    exr.name = "EXR_Output_Node"
    exr.directory = EXR
    exr.file_name = "data_"
    exr.format.file_format = "OPEN_EXR_MULTILAYER"
    exr.format.color_depth = "32"

    for i, name in enumerate(("Image", "Position", "IndexOB", "Depth")):
        if name in render_layers.outputs:
            slot = "Combined" if name == "Image" else name
            socket_type = 'RGBA' if name in ("Image", "Position") else 'FLOAT'

            exr.file_output_items.new(socket_type, slot)
            links.new(
                render_layers.outputs[name],
                exr.inputs[i]
            )



#Function to store camera locations, azimuth angle + number of concentric circles can be changed
def store_camera_locations():
    camera_locations = []

    M = Terrain.matrix_world
    M_inv = M.inverted()
    

    #Distance between each circle
    circle_distances = [8.0, 15.0, 22.0]  # Distances for each circle
    #Random jitter
    jitter = 1  # Maximum random jitter in meters

    min_dist = circle_distances[0] -jitter
    max_dist = circle_distances[-1] + jitter

    if MODE == "TRAINING":
        num_circles = 3
        num_locations_per_circle = 24

        for circle in range(num_circles):
            for azimuth in range(num_locations_per_circle):
                radius = circle_distances[circle] + random.uniform(-jitter, jitter)  # Add random jitter to the radius
                angle = (azimuth / num_locations_per_circle) * 360.0
                x = Lander.location.x + radius * math.cos(math.radians(angle))
                y = Lander.location.y + radius * math.sin(math.radians(angle))
                # Z coordinate set to CAMERA_HEIGHT above the "Moon_Terrain" at the current XY
                ray_start_world = M_inv @ Vector((x, y, 200))  # Start the ray above the terrain
                ray_dir_world = M_inv.to_3x3() @ Vector((0, 0, -1)).normalized()  # Cast the ray downwards
                hit, location, normal, index = Terrain.ray_cast(ray_start_world, ray_dir_world)

                if hit:
                    location = M @ location  # Transform back to world coordinates
                    z = location.z + CAMERA_HEIGHT  # Set Z to CAMERA_HEIGHT above the terrain
                    camera_locations.append((x, y, z, angle))

                else:
                    print(f"Skipped circle_{circle}_location_{azimuth}: No terrain found beneath it.")
                    z = CAMERA_HEIGHT  # Default to CAMERA_HEIGHT if no terrain is found
                    continue

    elif MODE == "TESTING":
        for i in range(NUM_TEST_FRAMES):
            radius = random.uniform(min_dist, max_dist)
            angle = random.uniform(0, 360)

            x = Lander.location.x + radius * math.cos(math.radians(angle))
            y = Lander.location.y + radius * math.sin(math.radians(angle))

            ray_start_world = M_inv @ Vector((x, y, 200))  # Start the ray above the terrain
            ray_dir_world = M_inv.to_3x3() @ Vector((0, 0, -1)).normalized()  # Cast the ray downwards
            hit, location, normal, index = Terrain.ray_cast(ray_start_world, ray_dir_world)

            if hit:
                location = M @ location  # Transform back to world coordinates
                z = location.z + CAMERA_HEIGHT  # Set Z to CAMERA_HEIGHT above the terrain
                camera_locations.append((x, y, z, angle))

            else:
                print(f"Skipped circle_{circle}_location_{azimuth}: No terrain found beneath it.")
                continue 

    return camera_locations



##Main 
def main():

    random.seed(67)

    scene = bpy.context.scene

    if os.path.isdir(MODE_DIR):
        shutil.rmtree(MODE_DIR)

    os.makedirs(RGB)
    os.makedirs(EXR)

    tag_lander(Lander)
    setup_gpu()

    camera_locations = store_camera_locations()

    poses_path = os.path.join(MODE_DIR, f"posev_{MODE.lower()}.csv")

    node_tree = scene.compositing_node_group
    exr_node = node_tree.nodes["EXR_Output_Node"]

    scene.render.use_file_extension = False

    with open(poses_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "frame",
            "x",
            "y",
            "z",
            "qw",
            "qx",
            "qy",
            "qz",
            "azimuth",
            "range"
        ])

        for frame, (x, y, z, angle) in enumerate(camera_locations):

            Camera.location = (x, y, z)

            scene.frame_set(frame)

            scene.render.filepath = os.path.join(RGB, f"frame_{frame:04d}.png")

            exr_node.file_name = f"frame_{frame:04d}.exr"

            bpy.ops.render.render(write_still=True)

            q = Camera.matrix_world.to_quaternion()

            writer.writerow([
                frame,
                x,
                y,
                z,
                q.w,
                q.x,
                q.y,
                q.z,
                angle,
                math.sqrt(
                    (x - Lander.location.x) ** 2 +
                    (y - Lander.location.y) ** 2
                )
            ])

            f.flush()

            bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


if __name__ == "__main__":
    main()