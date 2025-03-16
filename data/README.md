# KITTI Dataset Setup

This directory should contain the KITTI dataset for LiDAR obstacle detection.

## Download Instructions

1. Visit the [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/eval_object.php) website
2. Register for an account if you don't have one
3. Download the following files:
   - [Velodyne point clouds](http://www.cvlibs.net/download.php?file=data_object_velodyne.zip) (29 GB)
   - [Training labels](http://www.cvlibs.net/download.php?file=data_object_label_2.zip)
   - [Camera calibration matrices](http://www.cvlibs.net/download.php?file=data_object_calib.zip)
   - [Object development kit](http://www.cvlibs.net/download.php?file=devkit_object.zip)

   > **Note:** When prompted for "Purpose of download", briefly describe your intended usage of the dataset in 2-3 sentences (e.g., "This dataset will be used for research on LiDAR-based obstacle detection algorithms for autonomous vehicles. The project aims to improve detection accuracy in complex urban environments.")

## Directory Structure

After downloading and extracting, organize the data as follows:

```
data/
├── kitti/
│   ├── training/
│   │   ├── velodyne/       # Point cloud data (.bin files)
│   │   ├── label_2/        # Object labels
│   │   └── calib/          # Calibration files
│   └── testing/
│       ├── velodyne/       # Point cloud data (.bin files)
│       └── calib/          # Calibration files
└── processed/              # Will store preprocessed data
```

## Data Format

### Velodyne Point Clouds
- Binary files with extension `.bin`
- Each point is stored as 4 float values (x, y, z, intensity)
- Coordinates are in meters, in the Velodyne coordinate system

### Labels
- Text files with extension `.txt`
- Each line represents one object
- Format: type, truncation, occlusion, alpha, bbox(4), dimensions(3), location(3), rotation_y

### Calibration
- Text files with extension `.txt`
- Contains calibration matrices for converting between coordinate systems 