"""
File name: kalman_filter.py
Description: 2D Kalman Filter implementation for object tracking
Python Version: 3.7+
"""

import numpy as np


class KalmanFilter:
    """
    2D Kalman Filter for tracking object position and velocity.
    
    State vector: [x, y, vx, vy]
    - x, y: position coordinates
    - vx, vy: velocity components
    """
    
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        """
        Initialize the Kalman Filter.
        
        Args:
            dt: Time step (sampling time) for one cycle
            u_x: Acceleration in x-direction (control input)
            u_y: Acceleration in y-direction (control input)
            std_acc: Standard deviation of process noise (acceleration)
            x_std_meas: Standard deviation of measurement noise in x-direction
            y_std_meas: Standard deviation of measurement noise in y-direction
        """
        self.dt = dt
        
        # Control input vector u = [u_x, u_y]
        self.u = np.array([[u_x], [u_y]])
        
        # Initial state vector: [x0=0, y0=0, vx=0, vy=0]
        self.x = np.array([[0], [0], [0], [0]])
        
        # State transition matrix A (4x4)
        # x' = x + vx*dt
        # y' = y + vy*dt
        # vx' = vx
        # vy' = vy
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Control input matrix B (4x2)
        # Accounts for acceleration input
        self.B = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ])
        
        # Measurement matrix H (2x4)
        # We only observe position (x, y), not velocity
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance matrix Q (4x4)
        # Based on constant acceleration model
        # Q = σa² * [[dt⁴/4, 0, dt³/2, 0],
        #            [0, dt⁴/4, 0, dt³/2],
        #            [dt³/2, 0, dt², 0],
        #            [0, dt³/2, 0, dt²]]
        self.Q = np.array([
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2]
        ]) * std_acc**2
        
        # Measurement noise covariance matrix R (2x2)
        # Measurements are independent, so off-diagonal elements are 0
        # R = [[σx², 0],
        #      [0, σy²]]
        self.R = np.array([
            [x_std_meas**2, 0],
            [0, y_std_meas**2]
        ])
        
        # Initial prediction error covariance matrix P (4x4)
        # Initialize as identity matrix
        self.P = np.eye(4)
    
    def predict(self):
        """
        Predict the next state and error covariance.
        
        Performs time update:
        - x_k^- = A * x_{k-1} + B * u
        - P_k^- = A * P_{k-1} * A^T + Q
        
        Returns:
            Predicted position [x, y] as numpy array (2x1)
        """
        # Time update: predict state
        self.x = self.A @ self.x + self.B @ self.u
        
        # Time update: predict error covariance
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        # Return predicted position (first two elements of state vector)
        return self.x[:2]
    
    def update(self, z):
        """
        Update the state estimate with a new measurement.
        
        Performs measurement update:
        - S_k = H * P_k^- * H^T + R
        - K_k = P_k^- * H^T * S_k^-1
        - x_k = x_k^- + K_k * (z_k - H * x_k^-)
        - P_k = (I - K_k * H) * P_k^-
        
        Args:
            z: Measurement vector [x, y] (centroid coordinates)
        
        Returns:
            Updated position estimate [x, y] as numpy array (2x1)
        """
        # Convert measurement to column vector
        z = np.array([[z[0]], [z[1]]])
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.x = self.x + K @ (z - self.H @ self.x)
        
        # Update error covariance
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        # Return updated position estimate
        return self.x[:2]

