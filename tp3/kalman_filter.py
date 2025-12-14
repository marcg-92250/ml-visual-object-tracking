"""
File name: kalman_filter.py
Description: Kalman Filter adapted for bounding box tracking (centroid-based)
Python Version: 3.7+
"""

import numpy as np


class KalmanFilter:
    """
    2D Kalman Filter for tracking bounding box centroids.
    
    State vector: [cx, cy, vx, vy]
    - cx, cy: centroid coordinates
    - vx, vy: velocity components
    """
    
    def __init__(self, dt=1/30, std_acc=1, x_std_meas=1, y_std_meas=1):
        """
        Initialize the Kalman Filter for bounding box tracking.
        
        Args:
            dt: Time step (default: 1/30 for 30 fps)
            std_acc: Standard deviation of process noise (acceleration)
            x_std_meas: Standard deviation of measurement noise in x-direction
            y_std_meas: Standard deviation of measurement noise in y-direction
        """
        self.dt = dt
        
        # Initial state vector: [cx, cy, vx, vy]
        self.x = np.zeros((4, 1))
        
        # State transition matrix A (4x4)
        # cx' = cx + vx*dt
        # cy' = cy + vy*dt
        # vx' = vx
        # vy' = vy
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix H (2x4)
        # We only observe centroid position (cx, cy)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance matrix Q (4x4)
        self.Q = np.array([
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2]
        ]) * std_acc**2
        
        # Measurement noise covariance matrix R (2x2)
        self.R = np.array([
            [x_std_meas**2, 0],
            [0, y_std_meas**2]
        ])
        
        # Initial prediction error covariance matrix P (4x4)
        self.P = np.eye(4) * 100
    
    def init_state(self, cx, cy):
        """
        Initialize the state with a centroid position.
        
        Args:
            cx: Centroid x-coordinate
            cy: Centroid y-coordinate
        """
        self.x = np.array([[cx], [cy], [0], [0]])
    
    def predict(self):
        """
        Predict the next state and error covariance.
        
        Returns:
            Predicted centroid (cx, cy) as tuple
        """
        # Time update: predict state
        self.x = self.A @ self.x
        
        # Time update: predict error covariance
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        # Return predicted centroid
        return self.x[0, 0], self.x[1, 0]
    
    def update(self, cx, cy):
        """
        Update the state estimate with a new measurement.
        
        Args:
            cx: Measured centroid x-coordinate
            cy: Measured centroid y-coordinate
        
        Returns:
            Updated centroid (cx, cy) as tuple
        """
        # Convert measurement to column vector
        z = np.array([[cx], [cy]])
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.x = self.x + K @ (z - self.H @ self.x)
        
        # Update error covariance
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        # Return updated centroid
        return self.x[0, 0], self.x[1, 0]

