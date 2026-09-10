import bpy
import os.path

camera_locations = {}

#Scene variables
scene = bpy.context.scene
cam = scene.camera
output_path = bpy.path.abspath(scene.render.filepath)



