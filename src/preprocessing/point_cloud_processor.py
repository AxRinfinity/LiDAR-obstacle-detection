#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Point cloud preprocessing module for LiDAR data.
"""

import numpy as np
# import open3d as o3d
from sklearn.neighbors import NearestNeighbors

class PointCloudProcessor:
    """
    Class for preprocessing point cloud data from LiDAR sensors.
    
    This class handles:
    - Conversion between numpy arrays and Open3D point clouds
    - Voxel grid downsampling
    - Region of interest cropping
    - Statistical outlier removal
    - Radius outlier removal
    """
    
    def __init__(self, config):
        """
        Initialize the point cloud processor with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for preprocessing
        """
        self.config = config
        
        # Extract configuration parameters
        self.voxel_size = config.get('voxel_size', 0.1)
        
        # Region of interest parameters
        self.x_min = config.get('x_min', -40.0)
        self.x_max = config.get('x_max', 40.0)
        self.y_min = config.get('y_min', -20.0)
        self.y_max = config.get('y_max', 20.0)
        self.z_min = config.get('z_min', -2.5)
        self.z_max = config.get('z_max', 1.0)
        
        # Outlier removal parameters
        self.radius_outlier_radius = config.get('radius_outlier_radius', 0.5)
        self.radius_outlier_min_neighbors = config.get('radius_outlier_min_neighbors', 2)
    
    def process(self, point_cloud_data):
        """
        Process the input point cloud data.
        
        Args:
            point_cloud_data (numpy.ndarray): Input point cloud data (N, 4) where each row is [x, y, z, intensity]
            
        Returns:
            numpy.ndarray: Processed point cloud data (M, 4)
        """
        # Apply preprocessing steps directly on numpy arrays
        processed_data = self._crop_roi(point_cloud_data)
        processed_data = self._voxel_downsample(processed_data)
        processed_data = self._remove_outliers(processed_data)
        
        return processed_data
    
    def _crop_roi(self, point_cloud_data):
        """
        Crop the point cloud to the region of interest.
        
        Args:
            point_cloud_data (numpy.ndarray): Input point cloud data
            
        Returns:
            numpy.ndarray: Cropped point cloud data
        """
        # Extract XYZ coordinates
        x, y, z = point_cloud_data[:, 0], point_cloud_data[:, 1], point_cloud_data[:, 2]
        
        # Create mask for points within ROI
        mask = (
            (x >= self.x_min) & (x <= self.x_max) &
            (y >= self.y_min) & (y <= self.y_max) &
            (z >= self.z_min) & (z <= self.z_max)
        )
        
        # Apply mask
        return point_cloud_data[mask]
    
    def _voxel_downsample(self, point_cloud_data):
        """
        Downsample the point cloud using voxel grid.
        
        Args:
            point_cloud_data (numpy.ndarray): Input point cloud data
            
        Returns:
            numpy.ndarray: Downsampled point cloud data
        """
        # Get point coordinates
        points = point_cloud_data[:, :3]
        
        # Calculate voxel indices for each point
        voxel_indices = np.floor(points / self.voxel_size).astype(int)
        
        # Create a dictionary to store points in each voxel
        voxel_dict = {}
        
        # Group points by voxel
        for i in range(len(point_cloud_data)):
            voxel_key = tuple(voxel_indices[i])
            if voxel_key in voxel_dict:
                voxel_dict[voxel_key].append(i)
            else:
                voxel_dict[voxel_key] = [i]
        
        # Calculate centroid for each voxel
        downsampled_points = []
        
        for indices in voxel_dict.values():
            # Calculate mean of points in voxel
            centroid = np.mean(point_cloud_data[indices], axis=0)
            downsampled_points.append(centroid)
        
        return np.array(downsampled_points)
    
    def _remove_outliers(self, point_cloud_data):
        """
        Remove outliers from the point cloud.
        
        Args:
            point_cloud_data (numpy.ndarray): Input point cloud data
            
        Returns:
            numpy.ndarray: Filtered point cloud data
        """
        if len(point_cloud_data) < self.radius_outlier_min_neighbors + 1:
            return point_cloud_data
        
        # Get point coordinates
        points = point_cloud_data[:, :3]
        
        # Use nearest neighbors to find points within radius
        nn = NearestNeighbors(radius=self.radius_outlier_radius, algorithm='auto', n_jobs=-1)
        nn.fit(points)
        
        # Find neighbors within radius
        neighbors = nn.radius_neighbors(points, return_distance=False)
        
        # Count number of neighbors for each point
        neighbor_counts = np.array([len(n) for n in neighbors])
        
        # Keep points with enough neighbors
        mask = neighbor_counts >= self.radius_outlier_min_neighbors
        
        return point_cloud_data[mask] 