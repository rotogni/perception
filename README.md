# Stereo Visual Odometry 

This project implements a stereo visual odometry pipeline for estimating the trajectory of a rover and reconstructing a sparse 3D map of the environment using stereo images. The code is designed to work with datasets such as KITTI and Morocco (MADMAX), and leverages OpenCV for computer vision operations.

Screencast real time running on Apple MacBook Pro 13" M2 Max:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/s6yd67y4xYk/0.jpg)](https://www.youtube.com/watch?v=s6yd67y4xYk)

## Overview

The main program (`main.cpp`) processes stereo image sequences to:
- Detect and match features between left and right images.
- Triangulate 3D points from stereo correspondences.
- Track the camera pose over time using Perspective-n-Point (PnP) with RANSAC.
- Visualize stereo matches, 3D points, and the estimated trajectory.

## Main Components and Methods

### 1. Feature Detection and Description
- **FAST Detector**: Used to detect keypoints in both left and right images for efficiency.
- **ORB Descriptor**: Extracts binary descriptors for robust and fast feature matching.

### 2. Stereo Matching and Triangulation
- **kNN Matching with Lowe's Ratio Test**: Matches features between stereo pairs using k-nearest neighbors and applies Lowe's ratio test for robustness.
- **Epipolar Constraint**: Ensures matches are consistent with stereo geometry (vertical disparity check).
- **Triangulation**: Uses matched points to triangulate 3D landmarks in the camera frame, then transforms them to the world frame.

### 3. Pose Estimation (PnP)
- **Feature Matching to Landmarks**: Matches current frame features to previously triangulated 3D points.
- **PnP with RANSAC**: Estimates the camera pose by solving the Perspective-n-Point problem with RANSAC for outlier rejection.
- **Pose Update**: Updates the global pose (rotation and translation) of the camera.

### 4. Visualization
- **Stereo Matches**: Displays feature correspondences between left and right images.
- **3D Point Cloud**: Visualizes the reconstructed 3D points and the camera trajectory.

## Usage

### Compile
Compile `main.cpp` and dependencies using:

```bash
clang++ main.cpp pose_estimation.cpp visualization.cpp -o main -std=c++17 \
  -I/opt/homebrew/Cellar/opencv/4.11.0_1/include/opencv4 \
  -L/opt/homebrew/Cellar/opencv/4.11.0_1/lib \
  -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs \
  -lopencv_features2d -lopencv_calib3d -lopencv_viz
```

### Run

```bash
./main
```

## Datasets

- **KITTI**: Used for benchmarking visual odometry and 3D reconstruction.
- **Morocco (MADMAX)**: Used for rover navigation experiments.

## Credits

**KITTI Dataset:**
Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun (2013) Vision meets Robotics: The KITTI Dataset, International Journal of Robotics Research (IJRR)

**Morocco Dataset:**
Meyer, L., Smíšek, M., Fontan Villacampa, A., Oliva Maza, L., Medina, D., Schuster, M. J., Steidle, F., Vayugundla, M., Müller, M. G., Rebele, B., Wedler, A., & Triebel, R. (2021). The MADMAX data set for visual‐inertial rover navigation on Mars. Journal of Field Robotics, 1– 21. https://doi.org/10.1002/rob.22016
