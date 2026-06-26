import os
# Must be set before mujoco is imported to force headless rendering
os.environ["MUJOCO_GL"] = "egl" 
import mujoco
import numpy as np
import cv2

def test_mujoco_render():
    # Use the scene file which contains the robot, lighting, and cameras
    xml_path = "/home/tmainetucker/Repos/JEPA_Robotics/data/mujoco_assets/so101/scene.xml"
    
    if not os.path.exists(xml_path):
        print(f"Error: Could not find {xml_path}")
        return

    print("Loading MuJoCo model...")
    # Load model and data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print("Initializing simulation...")
    # Step simulation a few times to let physics settle
    for _ in range(10):
        mujoco.mj_step(model, data)

    print("Rendering 256x256 frame...")
    # Create renderer matching V-JEPA's exact input size
    renderer = mujoco.Renderer(model, height=256, width=256)
    
    # Refined Render 1: Rotated 90 deg, shifted to see pincers
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    
    # We shift lookat from X=0.15 to X=0.25 to center the pincers
    cam.lookat[:] = [0.25, 0.0, 0.0]  
    cam.distance = 0.75
    cam.elevation = -90  # Still straight down
    
    # Variation A: Azimuth 0
    cam.azimuth = 0
    renderer.update_scene(data, camera=cam)
    cv2.imwrite("/home/tmainetucker/Repos/JEPA_Robotics/data/mujoco_assets/render_1A_rotated_0.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    # Variation B: Azimuth 180 (opposite 90 deg rotation)
    cam.azimuth = 180
    renderer.update_scene(data, camera=cam)
    cv2.imwrite("/home/tmainetucker/Repos/JEPA_Robotics/data/mujoco_assets/render_1B_rotated_180.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    # Variation C: Azimuth 0, but slightly closer (zoomed in)
    cam.distance = 0.65
    cam.azimuth = 0
    renderer.update_scene(data, camera=cam)
    cv2.imwrite("/home/tmainetucker/Repos/JEPA_Robotics/data/mujoco_assets/render_1C_rotated_0_zoomed.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    # Variation D: Azimuth 180, slightly closer (zoomed in)
    cam.azimuth = 180
    renderer.update_scene(data, camera=cam)
    cv2.imwrite("/home/tmainetucker/Repos/JEPA_Robotics/data/mujoco_assets/render_1D_rotated_180_zoomed.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    print("Successfully generated 4 refined options in data/mujoco_assets/")

if __name__ == "__main__":
    test_mujoco_render()
