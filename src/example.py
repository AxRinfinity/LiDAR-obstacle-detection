#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to use the LiDAR obstacle detection pipeline.
"""

import os
import sys
import numpy as np
import yaml
import time
from pathlib import Path

from preprocessing.point_cloud_processor import PointCloudProcessor
from segmentation.ground_plane_segmenter import GroundPlaneSegmenter
from clustering.obstacle_clusterer import ObstacleClusterer
from tracking.obstacle_tracker import ObstacleTracker
from visualization.visualizer import Visualizer

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_point_cloud(file_path):
    """Load point cloud from KITTI .bin file."""
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

def process_single_frame(point_cloud, pipeline_components, visualize=True):
    """Process a single point cloud frame through the detection pipeline."""
    # Extract components
    processor, segmenter, clusterer, tracker, visualizer = pipeline_components
    
    # Preprocessing
    start_time = time.time()
    filtered_cloud = processor.process(point_cloud)
    preprocess_time = time.time() - start_time
    
    # Ground plane segmentation
    start_time = time.time()
    ground_points, obstacle_points = segmenter.segment(filtered_cloud)
    segmentation_time = time.time() - start_time
    
    # Obstacle clustering
    start_time = time.time()
    clusters = clusterer.cluster(obstacle_points)
    clustering_time = time.time() - start_time
    
    # Obstacle tracking
    start_time = time.time()
    tracked_obstacles = tracker.update(clusters)
    tracking_time = time.time() - start_time
    
    # Print timing information
    print(f"Preprocessing: {preprocess_time:.3f}s")
    print(f"Segmentation: {segmentation_time:.3f}s")
    print(f"Clustering: {clustering_time:.3f}s")
    print(f"Tracking: {tracking_time:.3f}s")
    print(f"Total: {preprocess_time + segmentation_time + clustering_time + tracking_time:.3f}s")
    
    # Print detection results
    print(f"Ground points: {ground_points.shape[0]}")
    print(f"Obstacle points: {obstacle_points.shape[0]}")
    print(f"Clusters: {len(clusters)}")
    print(f"Tracked obstacles: {len(tracked_obstacles)}")
    
    # Visualization
    if visualize:
        visualizer.visualize(
            original_cloud=point_cloud,
            ground_points=ground_points,
            obstacle_points=obstacle_points,
            clusters=clusters,
            tracked_obstacles=tracked_obstacles
        )
    
    return tracked_obstacles

def main():
    """Main function."""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python example.py <point_cloud_file.bin>")
        sys.exit(1)
    
    # Get point cloud file path
    point_cloud_path = Path(sys.argv[1])
    if not point_cloud_path.exists():
        print(f"Error: Point cloud file {point_cloud_path} does not exist.")
        sys.exit(1)
    
    # Load configuration
    config_path = Path("config/pipeline_config.yaml")
    if not config_path.exists():
        print(f"Error: Configuration file {config_path} does not exist.")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Initialize pipeline components
    processor = PointCloudProcessor(config['preprocessing'])
    segmenter = GroundPlaneSegmenter(config['segmentation'])
    clusterer = ObstacleClusterer(config['clustering'])
    tracker = ObstacleTracker(config['tracking'])
    visualizer = Visualizer(config['visualization'])
    
    pipeline_components = (processor, segmenter, clusterer, tracker, visualizer)
    
    # Load point cloud
    point_cloud = load_point_cloud(point_cloud_path)
    
    # Process point cloud
    tracked_obstacles = process_single_frame(point_cloud, pipeline_components)
    
    # Keep visualization window open until user closes it
    input("Press Enter to close visualization...")
    visualizer.close()

if __name__ == "__main__":
    main() 