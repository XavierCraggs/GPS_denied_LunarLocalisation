import numpy as np
import bpy
from mathutils import Vector
import os.path

## Constants
CAMERA_HEIGHT = 1.0  # Height of the camera above the terrain
Camera = bpy.data.objects['Rover_Camera']  
Terrain = bpy.data.objects['Moon_Terrain']  
Lander = bpy.data.objects['LAND_Base']

def cam_look_at(camera, target):
    track_to = camera.constraints.new(type='TRACK_TO')

    # Configure the constraint properties
    track_to.target = target
    track_to.track_axis = 'TRACK_NEGATIVE_Z'  # Blender cameras point down their local -Z axis
    track_to.up_axis = 'UP_Y'                 # Keeps the camera upright relative to its Y axis


#Start the camera tracking the lander (auto updates)
cam_look_at(Camera, Lander)



#Function to store camera locations, azimuth angle + number of concentric circles can be changed
def store_camera_locations():
    #Number of concentric circles
    num_circles = 3
    #Number of camera locations per circle
    num_locations_per_circle = 8
    #Distance between each circle
    distance_between_circles = 5.0

    for circle in range(num_circles):
        radius = (circle + 1) * distance_between_circles
        for azimuth in range(num_locations_per_circle):
            angle = (azimuth / num_locations_per_circle) * 360.0
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            # Z coordinate set to CAMERA_HEIGHT above the "Moon_Terrain" at the current XY
            ray_start_world = Vector((x, y, 10.0))  # Start the ray above the terrain
            ray_dir_world = Vector((0.0, 0.0, -1.0)) # Cast the ray downwards
            hit, location, normal, index, object, matrix = Terrain.ray_cast(ray_start_world, ray_dir_world)
            if hit:
                z = location.z + CAMERA_HEIGHT
            else:
                print(f"Skipped circle_{circle}_location_{azimuth}: No terrain found beneath it.")
                z = CAMERA_HEIGHT  # Default to CAMERA_HEIGHT if no terrain is found
                continue



            camera_locations[f"circle_{circle}_location_{azimuth}"] = (x, y, z, angle)

#Scene variables
scene = bpy.context.scene
cam = scene.camera
output_path = bpy.path.abspath(scene.render.filepath)

#Iterate through the list of camera locations, rendering a frame at each location and store the RGB + Ground Truth depth map


