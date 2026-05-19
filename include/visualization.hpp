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
    Visualization(const std::string& window_name);
    
    void initializeWindows();
    
    void showGroundTruthTrajectory(const std::vector<cv::Point3f>& gt_trajectory);
    
    void showStereoMatches(const cv::Mat& left_image,
                          const cv::Mat& right_image,
                          const std::vector<cv::KeyPoint>& left_keypoints,
                          const std::vector<cv::KeyPoint>& right_keypoints,
                          const std::vector<cv::DMatch>& matches);
    
    void showLegendWindow(size_t num_points, size_t num_poses, 
                         float min_depth, float max_depth);
    
    void updatePointCloud(const std::vector<cv::Point3f>& points3d,
                         const std::vector<cv::Point3f>& trajectory_points);
    
    void clearPointCloud();
    
    bool isWindowClosed() const;
    
    void cleanup();

private:
    std::string window_name;
    cv::viz::Viz3d viz_window;
    cv::viz::Viz3d trajectory_window;
    bool is_initialized;
    
    // Legend and statistics methods
    void addStaticLegend();
    void add2DLegend(cv::Mat& display_image, size_t num_points, size_t num_poses, 
                    float min_depth, float max_depth);
    void updateDynamicStats(size_t num_points, size_t num_poses, 
                           float min_depth, float max_depth);
};
