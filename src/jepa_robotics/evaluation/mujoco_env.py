import os
os.environ["MUJOCO_GL"] = "egl"
import mujoco
import numpy as np
import cv2
from jepa_robotics.data.kinematics import forward_kinematics

class SO100SimEnv:
    def __init__(self, xml_path="/home/tmainetucker/Repos/JEPA_Robotics/data/mujoco_assets/so101/scene.xml", image_size=256):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Could not find {xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=image_size, width=image_size)
        
        # -------------------------------------------------------------
        # Lock Camera to 1B Perspective (The Training Baseline)
        # -------------------------------------------------------------
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0.25, 0.0, 0.0]  
        self.cam.distance = 0.75
        self.cam.elevation = -90
        self.cam.azimuth = 180
        
    def reset(self, joint_angles=None):
        """Resets the robot to a specific joint configuration (or zeros)."""
        mujoco.mj_resetData(self.model, self.data)
        if joint_angles is not None:
            # SO100 has 6 joints
            for i in range(6):
                self.data.qpos[i] = joint_angles[i]
        mujoco.mj_forward(self.model, self.data)
        
    def step(self, action=None):
        """Steps physics forward by 1 tick using joint velocities or positions."""
        if action is not None:
            for i in range(6):
                self.data.ctrl[i] = action[i]
        mujoco.mj_step(self.model, self.data)
        
    def render(self):
        """Returns normalized 256x256x3 RGB float32 image."""
        self.renderer.update_scene(self.data, camera=self.cam)
        img = self.renderer.render()
        img = img.astype(np.float32) / 255.0
        return img
        
    def get_ground_truth_10d(self):
        """
        Uses the shared kinematics mapping to translate current MuJoCo 
        joint angles into the Golden 10D pose array.
        """
        # Grab current 6 DoF joint angles
        current_joints = np.array([self.data.qpos[i] for i in range(6)])
        
        # Expand dims for batch=1
        joints_batch = np.expand_dims(current_joints, axis=0)
        
        # Pass through the exact same DH table the ViT was trained against
        pose_10d = forward_kinematics(joints_batch, robot_type="so100")
        
        return pose_10d[0] # Return the 1D (10,) array

class SO100GhostEnv:
    """
    A secondary MuJoCo environment used purely for rendering the ViT's predictions 
    as a holographic 'Ghost Arm' overlaid on top of the real video feed.
    """
    def __init__(self, xml_path="/home/tmainetucker/Repos/JEPA_Robotics/data/mujoco_assets/so101/scene.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=256, width=256)
        
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0.25, 0.0, 0.0]  
        self.cam.distance = 0.75
        self.cam.elevation = -90
        self.cam.azimuth = 180
        
        # Render the empty background (robot at zero position) to use as a subtraction mask later
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam)
        # We need the background WITHOUT the robot. Since we can't easily delete it, 
        # we'll just move the robot out of sight for the mask!
        self.data.qpos[0] = 3.14 # Swing it 180 degrees away
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam)
        self.empty_bg = self.renderer.render().astype(np.float32) / 255.0
        
    def overlay_ghost(self, base_image, joint_angles):
        """
        Renders the ghost robot at the given joint angles, tints it holographic blue,
        and alpha-blends it perfectly over the provided base_image.
        """
        # 1. Pose the ghost
        for i in range(6):
            self.data.qpos[i] = joint_angles[i]
        mujoco.mj_forward(self.model, self.data)
        
        # 2. Render the ghost
        self.renderer.update_scene(self.data, camera=self.cam)
        ghost_img = self.renderer.render().astype(np.float32) / 255.0
        
        # 3. Extract foreground mask (Where does ghost_img differ from empty background?)
        diff = np.abs(ghost_img - self.empty_bg)
        mask = np.sum(diff, axis=-1) > 0.05
        
        # 4. Tint Ghost Holographic Blue
        blue_ghost = ghost_img.copy()
        blue_ghost[:, :, 0] *= 0.3 # Reduce Red
        blue_ghost[:, :, 1] *= 0.6 # Keep some Green for cyan feel
        blue_ghost[:, :, 2] = np.clip(blue_ghost[:, :, 2] * 2.0, 0, 1) # Boost Blue
        
        # 5. Alpha Blend only the masked pixels
        final_img = base_image.copy()
        final_img[mask] = (base_image[mask] * 0.4) + (blue_ghost[mask] * 0.6)
        
        return final_img
