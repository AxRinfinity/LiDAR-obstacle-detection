#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ground plane segmentation module for LiDAR data.
"""

import numpy as np
# import open3d as o3d
from sklearn.linear_model import RANSACRegressor

class GroundPlaneSegmenter:
    """
    Class for segmenting ground plane from obstacle points in LiDAR data.
    
    This class implements:
    - RANSAC-based plane segmentation
    - Normal-based filtering for ground plane identification
    - Separation of ground and obstacle points
    """
    
    def __init__(self, config):
        """
        Initialize the ground plane segmenter with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for segmentation
        """
        self.config = config
        
        # Extract RANSAC parameters
        self.distance_threshold = config.get('distance_threshold', 0.2)
        self.ransac_n = config.get('ransac_n', 3)
        self.num_iterations = config.get('num_iterations', 100)
        
        # Extract ground plane constraints
        self.ground_normal_threshold = config.get('ground_normal_threshold', 0.8)
        self.ground_z_max = config.get('ground_z_max', -1.0)
    
    def segment(self, point_cloud_data):
        """
        Segment the point cloud into ground and obstacle points.
        
        Args:
            point_cloud_data (numpy.ndarray): Input point cloud data (N, 4)
            
        Returns:
            tuple: (ground_points, obstacle_points) as numpy arrays
        """
        # Segment ground plane using RANSAC
        ground_indices, plane_model = self._segment_plane(point_cloud_data)
        
        # Validate ground plane
        is_valid_ground = self._validate_ground_plane(plane_model)
        
        if is_valid_ground:
            # Extract ground and obstacle points
            ground_points = point_cloud_data[ground_indices]
            obstacle_indices = np.ones(point_cloud_data.shape[0], dtype=bool)
            obstacle_indices[ground_indices] = False
            obstacle_points = point_cloud_data[obstacle_indices]
        else:
            # If no valid ground plane found, use height-based segmentation as fallback
            ground_points, obstacle_points = self._height_based_segmentation(point_cloud_data)
        
        return ground_points, obstacle_points
    
    def _segment_plane(self, point_cloud_data):
        """
        Segment the ground plane using RANSAC.
        
        Args:
            point_cloud_data (numpy.ndarray): Input point cloud data
            
        Returns:
            tuple: (ground_indices, plane_model)
        """
        # Extract XYZ coordinates
        points = point_cloud_data[:, :3]
        
        # Prepare data for RANSAC
        X = points[:, :2]  # x, y coordinates
        y = points[:, 2]   # z coordinates
        
        # Create RANSAC model for plane fitting
        ransac = RANSACRegressor(
            max_trials=self.num_iterations,
            residual_threshold=self.distance_threshold,
            random_state=42
        )
        
        # Fit model
        try:
            ransac.fit(X, y)
            
            # Get inlier indices
            inlier_mask = ransac.inlier_mask_
            
            # Calculate plane model (ax + by + c = z)
            a, b = ransac.estimator_.coef_
            c = ransac.estimator_.intercept_
            
            # Convert to standard plane equation (ax + by + cz + d = 0)
            plane_model = np.array([a, b, -1, c])
            plane_model = plane_model / np.linalg.norm(plane_model[:3])
            
            return np.where(inlier_mask)[0], plane_model
            
        except Exception as e:
            print(f"RANSAC failed: {e}")
            # Return empty indices and default plane model
            return np.array([], dtype=int), np.array([0, 0, 1, 0])
    
    def _validate_ground_plane(self, plane_model):
        """
        Validate if the detected plane is a ground plane.
        
        Args:
            plane_model (numpy.ndarray): Plane model [a, b, c, d] where ax + by + cz + d = 0
            
        Returns:
            bool: True if the plane is a valid ground plane
        """
        # Extract plane normal
        a, b, c, _ = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        # Check if normal is pointing upward (approximately parallel to z-axis)
        # The ground normal should be close to [0, 0, 1] or [0, 0, -1]
        up_vector = np.array([0, 0, 1])
        dot_product = np.abs(np.dot(normal, up_vector))
        
        return dot_product > self.ground_normal_threshold
    
    def _height_based_segmentation(self, point_cloud_data):
        """
        Fallback method: Segment ground based on height (z-coordinate).
        
        Args:
            point_cloud_data (numpy.ndarray): Input point cloud data (N, 4)
            
        Returns:
            tuple: (ground_points, obstacle_points) as numpy arrays
        """
        # Use z-coordinate for segmentation
        z = point_cloud_data[:, 2]
        
        # Points below threshold are considered ground
        ground_indices = z < self.ground_z_max
        obstacle_indices = ~ground_indices
        
        ground_points = point_cloud_data[ground_indices]
        obstacle_points = point_cloud_data[obstacle_indices]
        
        return ground_points, obstacle_points
    
    def _refine_ground_segmentation(self, point_cloud_data, initial_ground_indices):
        """
        Refine ground segmentation using multiple iterations.
        
        Args:
            point_cloud_data (numpy.ndarray): Input point cloud data (N, 4)
            initial_ground_indices (numpy.ndarray): Initial ground point indices
            
        Returns:
            numpy.ndarray: Refined ground point indices
        """
        # Extract initial ground points
        ground_points = point_cloud_data[initial_ground_indices]
        
        # Convert to Open3D point cloud
        pcd = self._numpy_to_o3d(ground_points)
        
        # Fit a plane to the ground points
        plane_model, _ = pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations
        )
        
        # Extract plane parameters
        a, b, c, d = plane_model
        
        # Calculate distance of all points to the plane
        points = point_cloud_data[:, :3]
        distances = np.abs(np.dot(points, [a, b, c]) + d) / np.sqrt(a**2 + b**2 + c**2)
        
        # Points close to the plane are considered ground
        refined_ground_indices = distances < self.distance_threshold
        
        return refined_ground_indices 