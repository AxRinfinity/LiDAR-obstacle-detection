#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Obstacle tracking module for LiDAR data.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import uuid

class KalmanFilter:
    """
    Kalman Filter for tracking obstacles in 3D space.
    
    This implements a constant velocity model for tracking objects.
    """
    
    def __init__(self, process_noise_scale=0.1, measurement_noise_scale=0.1):
        """
        Initialize the Kalman filter.
        
        Args:
            process_noise_scale (float): Scale factor for process noise
            measurement_noise_scale (float): Scale factor for measurement noise
        """
        # State dimension: [x, y, z, vx, vy, vz, width, length, height]
        self.state_dim = 9
        
        # Measurement dimension: [x, y, z, width, length, height]
        self.meas_dim = 6
        
        # Initialize state
        self.x = np.zeros((self.state_dim, 1))
        
        # Initialize state covariance
        self.P = np.eye(self.state_dim)
        
        # Process noise scale
        self.Q_scale = process_noise_scale
        
        # Measurement noise scale
        self.R_scale = measurement_noise_scale
        
        # Initialize process noise covariance
        self.Q = np.eye(self.state_dim) * self.Q_scale
        
        # Initialize measurement noise covariance
        self.R = np.eye(self.meas_dim) * self.R_scale
        
        # Initialize state transition matrix (constant velocity model)
        self.F = np.eye(self.state_dim)
        # Position update with velocity
        self.F[0, 3] = 1.0  # x += vx
        self.F[1, 4] = 1.0  # y += vy
        self.F[2, 5] = 1.0  # z += vz
        
        # Initialize measurement matrix
        self.H = np.zeros((self.meas_dim, self.state_dim))
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y
        self.H[2, 2] = 1.0  # z
        self.H[3, 6] = 1.0  # width
        self.H[4, 7] = 1.0  # length
        self.H[5, 8] = 1.0  # height
    
    def predict(self, dt=1.0):
        """
        Predict the state forward by time dt.
        
        Args:
            dt (float): Time step
            
        Returns:
            numpy.ndarray: Predicted state
        """
        # Update state transition matrix for current dt
        self.F[0, 3] = dt  # x += vx * dt
        self.F[1, 4] = dt  # y += vy * dt
        self.F[2, 5] = dt  # z += vz * dt
        
        # Predict state
        self.x = self.F @ self.x
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x
    
    def update(self, z):
        """
        Update the state with a new measurement.
        
        Args:
            z (numpy.ndarray): Measurement vector [x, y, z, width, length, height]
            
        Returns:
            numpy.ndarray: Updated state
        """
        # Reshape measurement to column vector
        z = z.reshape((self.meas_dim, 1))
        
        # Calculate innovation (measurement residual)
        y = z - self.H @ self.x
        
        # Calculate innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Calculate Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x
    
    def get_state(self):
        """
        Get the current state.
        
        Returns:
            dict: Current state as a dictionary
        """
        return {
            'position': self.x[:3, 0],
            'velocity': self.x[3:6, 0],
            'size': self.x[6:9, 0]
        }


class Track:
    """
    Class representing a tracked obstacle.
    """
    
    def __init__(self, cluster, track_id=None, process_noise_scale=0.1, measurement_noise_scale=0.1):
        """
        Initialize a new track.
        
        Args:
            cluster (dict): Cluster object from ObstacleClusterer
            track_id (str): Unique track ID (generated if None)
            process_noise_scale (float): Scale factor for process noise
            measurement_noise_scale (float): Scale factor for measurement noise
        """
        # Generate unique ID if not provided
        self.id = track_id if track_id is not None else str(uuid.uuid4())
        
        # Initialize Kalman filter
        self.kf = KalmanFilter(process_noise_scale, measurement_noise_scale)
        
        # Extract initial state from cluster
        centroid = cluster['centroid']
        bbox = cluster['bbox']
        
        # Initial measurement: [x, y, z, width, length, height]
        z = np.array([
            centroid[0],
            centroid[1],
            centroid[2],
            bbox['size'][0],
            bbox['size'][1],
            bbox['size'][2]
        ])
        
        # Initialize state with zero velocity
        x = np.zeros(9)
        x[:3] = centroid
        x[6:9] = bbox['size']
        
        # Set initial state
        self.kf.x = x.reshape((9, 1))
        
        # Initialize track attributes
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.color = cluster['color']
        self.history = [centroid]
        self.orientation = bbox['orientation']
        self.last_cluster = cluster
    
    def predict(self, dt=1.0):
        """
        Predict the track state forward by time dt.
        
        Args:
            dt (float): Time step
            
        Returns:
            dict: Predicted state
        """
        self.kf.predict(dt)
        self.age += 1
        self.time_since_update += 1
        
        # Get current state
        state = self.kf.get_state()
        
        # Add to history
        self.history.append(state['position'])
        
        return state
    
    def update(self, cluster):
        """
        Update the track with a new cluster measurement.
        
        Args:
            cluster (dict): Cluster object from ObstacleClusterer
            
        Returns:
            dict: Updated state
        """
        # Extract measurement from cluster
        centroid = cluster['centroid']
        bbox = cluster['bbox']
        
        # Measurement: [x, y, z, width, length, height]
        z = np.array([
            centroid[0],
            centroid[1],
            centroid[2],
            bbox['size'][0],
            bbox['size'][1],
            bbox['size'][2]
        ])
        
        # Update Kalman filter
        self.kf.update(z)
        
        # Update track attributes
        self.hits += 1
        self.time_since_update = 0
        self.orientation = bbox['orientation']
        self.last_cluster = cluster
        
        # Get current state
        state = self.kf.get_state()
        
        return state
    
    def get_state(self):
        """
        Get the current track state.
        
        Returns:
            dict: Current track state
        """
        state = self.kf.get_state()
        
        return {
            'id': self.id,
            'position': state['position'],
            'velocity': state['velocity'],
            'size': state['size'],
            'orientation': self.orientation,
            'age': self.age,
            'hits': self.hits,
            'time_since_update': self.time_since_update,
            'color': self.color,
            'history': self.history,
            'last_cluster': self.last_cluster
        }


class ObstacleTracker:
    """
    Class for tracking obstacles across frames.
    
    This class implements:
    - Track initialization and management
    - Data association between tracks and new detections
    - Kalman filtering for state estimation
    """
    
    def __init__(self, config):
        """
        Initialize the obstacle tracker with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for tracking
        """
        self.config = config
        
        # Extract Kalman filter parameters
        self.process_noise_scale = config.get('process_noise_scale', 0.1)
        self.measurement_noise_scale = config.get('measurement_noise_scale', 0.1)
        
        # Extract association parameters
        self.max_distance = config.get('max_distance', 2.0)
        
        # Extract track management parameters
        self.min_hits_to_initialize = config.get('min_hits_to_initialize', 3)
        self.max_age = config.get('max_age', 5)
        
        # Initialize tracks
        self.tracks = []
        
        # Initialize track ID counter
        self.next_track_id = 0
    
    def update(self, clusters, dt=1.0):
        """
        Update tracks with new cluster detections.
        
        Args:
            clusters (list): List of cluster objects from ObstacleClusterer
            dt (float): Time step since last update
            
        Returns:
            list: List of updated tracks
        """
        # Predict all tracks forward
        for track in self.tracks:
            track.predict(dt)
        
        # Associate detections with tracks
        assignments, unassigned_tracks, unassigned_clusters = self._associate_detections_to_tracks(clusters)
        
        # Update assigned tracks
        for track_idx, cluster_idx in assignments:
            self.tracks[track_idx].update(clusters[cluster_idx])
        
        # Create new tracks for unassigned detections
        for cluster_idx in unassigned_clusters:
            self._create_new_track(clusters[cluster_idx])
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]
        
        # Get confirmed tracks (tracks with enough hits)
        confirmed_tracks = [track for track in self.tracks if track.hits >= self.min_hits_to_initialize]
        
        # Return track states
        return [track.get_state() for track in confirmed_tracks]
    
    def _associate_detections_to_tracks(self, clusters):
        """
        Associate detections with existing tracks using Hungarian algorithm.
        
        Args:
            clusters (list): List of cluster objects
            
        Returns:
            tuple: (assignments, unassigned_tracks, unassigned_clusters)
        """
        # If no tracks or no clusters, return empty assignments
        if len(self.tracks) == 0 or len(clusters) == 0:
            return [], list(range(len(self.tracks))), list(range(len(clusters)))
        
        # Calculate cost matrix
        cost_matrix = np.zeros((len(self.tracks), len(clusters)))
        
        for i, track in enumerate(self.tracks):
            track_pos = track.kf.x[:3, 0]
            
            for j, cluster in enumerate(clusters):
                cluster_pos = cluster['centroid']
                
                # Calculate Euclidean distance
                distance = np.linalg.norm(track_pos - cluster_pos)
                
                # Set cost to distance if within max_distance, otherwise to a large value
                if distance <= self.max_distance:
                    cost_matrix[i, j] = distance
                else:
                    cost_matrix[i, j] = 1000000  # A large value
        
        # Solve assignment problem using Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Create assignments list
        assignments = []
        unassigned_tracks = list(range(len(self.tracks)))
        unassigned_clusters = list(range(len(clusters)))
        
        # Filter assignments based on max_distance
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= self.max_distance:
                assignments.append((row, col))
                unassigned_tracks.remove(row)
                unassigned_clusters.remove(col)
        
        return assignments, unassigned_tracks, unassigned_clusters
    
    def _create_new_track(self, cluster):
        """
        Create a new track from a cluster.
        
        Args:
            cluster (dict): Cluster object
        """
        # Create new track
        track = Track(
            cluster,
            track_id=f"track_{self.next_track_id}",
            process_noise_scale=self.process_noise_scale,
            measurement_noise_scale=self.measurement_noise_scale
        )
        
        # Add to tracks list
        self.tracks.append(track)
        
        # Increment track ID counter
        self.next_track_id += 1 