#pragma once
#include "types.hpp"  
#include <opencv2/opencv.hpp>
#include <vector>


class PoseEstimation {
public:
    PoseEstimation();
    
    // Initialize 3D points from stereo pair
    void initialize3D(const cv::Mat& left_image,
                      const cv::Mat& right_image,
                      std::vector<cv::Point3f>& points3d,
                      std::vector<cv::Mat>& points_3d_descriptors,
                      std::vector<size_t>& points_3d_valid_indices,
                      std::vector<cv::KeyPoint>& left_keypoints,
                      std::vector<cv::KeyPoint>& right_keypoints,
                      std::vector<cv::DMatch>& matches,
                      bool verbose = false);
    
    // Estimate pose using PnP
    void PnP(const cv::Mat& left_image,
             std::vector<cv::Point3f>& points3d,
             std::vector<cv::Mat>& points_3d_descriptors,
             std::vector<size_t>& points_3d_valid_indices,
             bool verbose = false);
    
    // Get current pose
    void getCurrentPose(cv::Mat& R_out, cv::Mat& t_out) const {
        R_out = R.clone();
        t_out = t.clone();
    }
    
    // Set current pose
    void setCurrentPose(const cv::Mat& R_in, const cv::Mat& t_in) {
        R = R_in.clone();
        t = t_in.clone();
    }

private:
    // Camera parameters (KITTI dataset defaults)
    double focal_length = 718.856;
    cv::Point2d principal_point = cv::Point2d(607.1928, 185.2157);
    double baseline = 0.54; // meters
    
    // Feature detection and matching
    cv::Ptr<cv::FastFeatureDetector> fast_detector;
    cv::Ptr<cv::ORB> descriptor_extractor;
    cv::Ptr<cv::BFMatcher> matcher;
    cv::Ptr<cv::BFMatcher> knn_matcher;
    
    // Current pose (camera-to-world transformation)
    cv::Mat R; // Rotation matrix (3x3)
    cv::Mat t; // Translation vector (3x1)
};
