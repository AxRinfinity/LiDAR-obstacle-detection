#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization module for LiDAR data and obstacle detection results.
"""

import numpy as np
# import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import time

class Visualizer:
    """
    Class for visualizing point clouds and obstacle detection results.
    
    This class implements:
    - Point cloud visualization
    - Ground plane visualization
    - Obstacle cluster visualization
    - Bounding box visualization
    - Track visualization
    """
    
    def __init__(self, config):
        """
        Initialize the visualizer with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for visualization
        """
        self.config = config
        
        # Extract point cloud colors
        self.original_color = config.get('original_color', [0.5, 0.5, 0.5])
        self.ground_color = config.get('ground_color', [0.0, 1.0, 0.0])
        self.obstacle_color = config.get('obstacle_color', [1.0, 0.0, 0.0])
        
        # Extract cluster visualization parameters
        self.use_cluster_colors = config.get('use_cluster_colors', True)
        
        # Extract bounding box visualization parameters
        self.show_bounding_boxes = config.get('show_bounding_boxes', True)
        self.box_line_width = config.get('box_line_width', 2.0)
        
        # Extract display settings
        self.point_size = config.get('point_size', 2.0)
        self.background_color = config.get('background_color', [0.0, 0.0, 0.0])
        self.window_width = config.get('window_width', 1280)
        self.window_height = config.get('window_height', 720)
        
        # Initialize figure
        self.fig = None
        self.ax = None
    
    def visualize(self, original_cloud=None, ground_points=None, obstacle_points=None, 
                 clusters=None, tracked_obstacles=None):
        """
        Visualize point clouds and detection results.
        
        Args:
            original_cloud (numpy.ndarray): Original point cloud data (N, 4)
            ground_points (numpy.ndarray): Ground plane points (M, 4)
            obstacle_points (numpy.ndarray): Obstacle points (P, 4)
            clusters (list): List of cluster objects
            tracked_obstacles (list): List of tracked obstacle objects
        """
        # Create figure and axis
        self.fig = plt.figure(figsize=(self.window_width/100, self.window_height/100), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set background color
        self.fig.patch.set_facecolor(self.background_color)
        self.ax.set_facecolor(self.background_color)
        
        # Set axis labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Set title
        self.ax.set_title('LiDAR Point Cloud Visualization')
        
        # Visualize original point cloud if provided
        if original_cloud is not None:
            self._visualize_point_cloud(original_cloud, self.original_color, 'Original Points')
        
        # Visualize ground points if provided
        if ground_points is not None:
            self._visualize_point_cloud(ground_points, self.ground_color, 'Ground Points')
        
        # Visualize obstacle points if provided
        if obstacle_points is not None and clusters is None:
            self._visualize_point_cloud(obstacle_points, self.obstacle_color, 'Obstacle Points')
        
        # Visualize clusters if provided
        if clusters is not None:
            self._visualize_clusters(clusters)
        
        # Visualize tracked obstacles if provided
        if tracked_obstacles is not None:
            self._visualize_tracked_obstacles(tracked_obstacles)
        
        # Set equal aspect ratio
        self.ax.set_aspect('auto')
        
        # Add legend
        self.ax.legend()
        
        # Show plot
        plt.tight_layout()
        plt.show()
    
    def _visualize_point_cloud(self, points, color, label, max_points=5000):
        """
        Visualize a point cloud.
        
        Args:
            points (numpy.ndarray): Point cloud data (N, 4)
            color (list): RGB color for the point cloud
            label (str): Label for the legend
            max_points (int): Maximum number of points to visualize
        """
        # Subsample points if there are too many
        if points.shape[0] > max_points:
            indices = np.random.choice(points.shape[0], max_points, replace=False)
            points = points[indices]
        
        # Extract XYZ coordinates
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # Plot points
        self.ax.scatter(x, y, z, c=[color], s=self.point_size, alpha=0.5, label=label)
    
    def _visualize_clusters(self, clusters):
        """
        Visualize clusters.
        
        Args:
            clusters (list): List of cluster objects
        """
        for i, cluster in enumerate(clusters):
            # Get cluster points
            points = cluster['points']
            
            # Get cluster color
            if self.use_cluster_colors:
                color = cluster['color']
            else:
                color = self.obstacle_color
            
            # Visualize cluster points
            self._visualize_point_cloud(
                points, color, f'Cluster {i+1}', max_points=100
            )
            
            # Visualize bounding box if enabled
            if self.show_bounding_boxes:
                self._visualize_bounding_box(cluster['bbox'], color)
    
    def _visualize_tracked_obstacles(self, tracked_obstacles):
        """
        Visualize tracked obstacles.
        
        Args:
            tracked_obstacles (list): List of tracked obstacle objects
        """
        for i, track in enumerate(tracked_obstacles):
            # Get track color
            color = track['color']
            
            # Visualize bounding box if enabled
            if self.show_bounding_boxes:
                # Create bounding box from track state
                bbox_params = {
                    'center': track['position'],
                    'size': track['size'],
                    'orientation': track['orientation'],
                    'type': 'oriented' if 'orientation' in track else 'axis_aligned'
                }
                
                self._visualize_bounding_box(bbox_params, color)
            
            # Visualize track history if available
            if 'history' in track and len(track['history']) > 1:
                self._visualize_track_history(track['history'], color, track['id'])
    
    def _visualize_bounding_box(self, bbox, color):
        """
        Visualize a bounding box.
        
        Args:
            bbox (dict): Bounding box parameters
            color (list): RGB color for the bounding box
        """
        # Get bounding box parameters
        center = bbox['center']
        size = bbox['size']
        
        # Calculate corners
        half_size = size / 2
        
        # For axis-aligned bounding box
        if bbox['type'] == 'axis_aligned':
            # Calculate corners
            corners = np.array([
                [center[0] - half_size[0], center[1] - half_size[1], center[2] - half_size[2]],
                [center[0] + half_size[0], center[1] - half_size[1], center[2] - half_size[2]],
                [center[0] + half_size[0], center[1] + half_size[1], center[2] - half_size[2]],
                [center[0] - half_size[0], center[1] + half_size[1], center[2] - half_size[2]],
                [center[0] - half_size[0], center[1] - half_size[1], center[2] + half_size[2]],
                [center[0] + half_size[0], center[1] - half_size[1], center[2] + half_size[2]],
                [center[0] + half_size[0], center[1] + half_size[1], center[2] + half_size[2]],
                [center[0] - half_size[0], center[1] + half_size[1], center[2] + half_size[2]]
            ])
        else:
            # For oriented bounding box
            orientation = bbox['orientation']
            
            # Calculate corners for unit cube
            unit_corners = np.array([
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1]
            ]) * 0.5
            
            # Scale corners
            scaled_corners = unit_corners * size.reshape(1, 3)
            
            # Rotate corners
            rotated_corners = np.dot(scaled_corners, orientation.T)
            
            # Translate corners
            corners = rotated_corners + center.reshape(1, 3)
        
        # Define edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        
        # Plot edges
        for start, end in edges:
            self.ax.plot(
                [corners[start, 0], corners[end, 0]],
                [corners[start, 1], corners[end, 1]],
                [corners[start, 2], corners[end, 2]],
                color=color, linewidth=self.box_line_width
            )
    
    def _visualize_track_history(self, history, color, track_id):
        """
        Visualize track history.
        
        Args:
            history (list): List of track positions
            color (list): RGB color for the track
            track_id (str): Track ID
        """
        # Convert history to numpy array
        history = np.array(history)
        
        # Plot track history
        self.ax.plot(
            history[:, 0], history[:, 1], history[:, 2],
            color=color, linewidth=2, label=f'Track {track_id}'
        )
    
    def close(self):
        """Close the visualization."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None 