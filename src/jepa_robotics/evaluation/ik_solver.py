import numpy as np
from scipy.optimize import minimize
from jepa_robotics.data.kinematics import forward_kinematics

class IKSolver:
    def __init__(self, robot_type="so100"):
        self.robot_type = robot_type
        
    def solve(self, target_10d, initial_guess=None):
        """
        Solves Inverse Kinematics by running optimization against our native 
        Forward Kinematics mathematical model.
        
        Args:
            target_10d: (10,) numpy array [X, Y, Z, R6D..., Gripper]
            initial_guess: (6,) numpy array of joint angles. If None, uses zeros.
            
        Returns:
            (6,) numpy array of joint angles that produce the target_10d.
        """
        if initial_guess is None:
            initial_guess = np.zeros(6)
            
        # The objective function to minimize
        def objective(joints):
            joints_batch = np.expand_dims(joints, axis=0)
            pred_10d = forward_kinematics(joints_batch, robot_type=self.robot_type)[0]
            
            # Position MSE
            pos_err = np.mean((pred_10d[:3] - target_10d[:3])**2)
            # Rotation 6D MSE
            rot_err = np.mean((pred_10d[3:9] - target_10d[3:9])**2)
            # Gripper MSE
            grip_err = np.mean((pred_10d[9:] - target_10d[9:])**2)
            
            # We care heavily about position and rotation.
            return (pos_err * 10.0) + (rot_err * 1.0) + (grip_err * 0.1)
            
        # SO100 Joint Limits (approximate radians)
        bounds = [
            (-1.9, 1.9),   # Shoulder Pan
            (-1.74, 1.74), # Shoulder Lift
            (-1.69, 1.69), # Elbow Flex
            (-1.65, 1.65), # Wrist Flex
            (-2.8, 2.8),   # Wrist Roll
            (-0.17, 1.74)  # Gripper
        ]
        
        # SLSQP is excellent for bounded constraint problems
        res = minimize(
            objective, 
            initial_guess, 
            method='SLSQP', 
            bounds=bounds,
            options={'ftol': 1e-4, 'maxiter': 50}
        )
        
        return res.x
