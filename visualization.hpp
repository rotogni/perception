#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include "types.hpp"

class Visualization {
public:
    Visualization(const std::string& window_name = "3D Point Cloud");
    void initializeWindows();
    void showStereoMatches(const cv::Mat& left_image,
                          const cv::Mat& right_image,
                          const std::vector<cv::KeyPoint>& left_keypoints,
                          const std::vector<cv::KeyPoint>& right_keypoints,
                          const std::vector<cv::DMatch>& matches);
    void updatePointCloud(const std::vector<cv::Point3f>& points3d,
                         const std::vector<cv::Point3f>& trajectory_points = std::vector<cv::Point3f>());
    void clearPointCloud();
    bool isWindowClosed() const;
    void cleanup();
    void showLegendWindow();
    // Show ground truth trajectory as a green polyline
    void showGroundTruthTrajectory(const std::vector<cv::Point3f>& gt_trajectory);
    
private:
    cv::viz::Viz3d viz_window;
    std::string window_name;
    std::string trajectory_window;
    bool is_initialized;
};
