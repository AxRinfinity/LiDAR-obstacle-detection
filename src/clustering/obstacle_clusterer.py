#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Obstacle clustering module for LiDAR data.
"""

import numpy as np
# Pašaliname open3d importą ir naudojame tik sklearn
# import open3d as o3d
from sklearn.cluster import DBSCAN
import random

class ObstacleClusterer:
    """
    Class for clustering obstacle points into distinct objects.
    
    This class implements:
    - DBSCAN clustering for obstacle points
    - Cluster filtering based on size
    - Bounding box generation for clusters
    """
    
    def __init__(self, config):
        """
        Initialize the obstacle clusterer with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for clustering
        """
        self.config = config
        
        # Extract DBSCAN parameters
        self.eps = config.get('eps', 0.5)
        self.min_points = config.get('min_points', 10)
        
        # Extract cluster filtering parameters
        self.min_cluster_size = config.get('min_cluster_size', 10)
        self.max_cluster_size = config.get('max_cluster_size', 10000)
    
    def cluster(self, obstacle_points):
        """
        Cluster obstacle points into distinct objects.
        
        Args:
            obstacle_points (numpy.ndarray): Obstacle point cloud data (N, 4)
            
        Returns:
            list: List of clusters, where each cluster is a dict containing:
                - 'points': numpy.ndarray of points in the cluster
                - 'centroid': numpy.ndarray of cluster centroid
                - 'bbox': dict with 'center', 'size', 'orientation'
                - 'color': numpy.ndarray of RGB color for visualization
        """
        # Check if there are enough points to cluster
        if obstacle_points.shape[0] < self.min_points:
            return []
        
        # Extract XYZ coordinates
        xyz = obstacle_points[:, :3]
        
        # Perform DBSCAN clustering
        labels = self._dbscan_clustering(xyz)
        
        # Process clusters
        clusters = self._process_clusters(obstacle_points, labels)
        
        return clusters
    
    def _dbscan_clustering(self, points):
        """
        Perform DBSCAN clustering on points.
        
        Args:
            points (numpy.ndarray): XYZ coordinates of points (N, 3)
            
        Returns:
            numpy.ndarray: Cluster labels for each point
        """
        # Create DBSCAN clusterer
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_points,
            metric='euclidean',
            algorithm='ball_tree',
            n_jobs=-1
        )
        
        # Perform clustering
        labels = dbscan.fit_predict(points)
        
        return labels
    
    def _process_clusters(self, points, labels):
        """
        Process DBSCAN clustering results into cluster objects.
        
        Args:
            points (numpy.ndarray): Point cloud data (N, 4)
            labels (numpy.ndarray): Cluster labels for each point
            
        Returns:
            list: List of cluster objects
        """
        # Get unique labels (excluding noise label -1)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        clusters = []
        
        # Process each cluster
        for label in unique_labels:
            # Get points in this cluster
            cluster_indices = labels == label
            cluster_points = points[cluster_indices]
            
            # Filter clusters by size
            if (cluster_points.shape[0] < self.min_cluster_size or
                cluster_points.shape[0] > self.max_cluster_size):
                continue
            
            # Calculate cluster centroid
            centroid = np.mean(cluster_points[:, :3], axis=0)
            
            # Generate bounding box
            bbox = self._generate_bounding_box(cluster_points)
            
            # Generate random color for visualization
            color = self._generate_random_color()
            
            # Create cluster object
            cluster = {
                'points': cluster_points,
                'centroid': centroid,
                'bbox': bbox,
                'color': color,
                'size': cluster_points.shape[0]
            }
            
            clusters.append(cluster)
        
        return clusters
    
    def _generate_bounding_box(self, cluster_points):
        """
        Generate oriented bounding box for a cluster.
        
        Args:
            cluster_points (numpy.ndarray): Points in the cluster (N, 4)
            
        Returns:
            dict: Bounding box parameters
        """
        # Pakeičiame open3d bounding box generavimą į paprastą axis-aligned bounding box
        # Apskaičiuojame min ir max taškus
        min_bound = np.min(cluster_points[:, :3], axis=0)
        max_bound = np.max(cluster_points[:, :3], axis=0)
        
        # Apskaičiuojame centrą ir dydį
        center = (min_bound + max_bound) / 2
        size = max_bound - min_bound
        
        # Sukuriame bounding box objektą
        bbox = {
            'center': center,
            'size': size,
            'orientation': np.eye(3),  # Identity matrix for axis-aligned
            'type': 'axis_aligned'
        }
        
        return bbox
    
    def _generate_random_color(self):
        """
        Generate a random RGB color for cluster visualization.
        
        Returns:
            numpy.ndarray: RGB color values
        """
        # Generate bright, distinguishable colors
        h = random.random()  # Hue
        s = 0.7 + random.random() * 0.3  # Saturation
        v = 0.7 + random.random() * 0.3  # Value
        
        # Convert HSV to RGB
        h_i = int(h * 6)
        f = h * 6 - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        if h_i == 0:
            r, g, b = v, t, p
        elif h_i == 1:
            r, g, b = q, v, p
        elif h_i == 2:
            r, g, b = p, v, t
        elif h_i == 3:
            r, g, b = p, q, v
        elif h_i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        
        return np.array([r, g, b]) 