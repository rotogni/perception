#pragma once
#include "types.hpp"  
#include <opencv2/opencv.hpp>
#include <vector>


class PoseEstimation {
public:
    PoseEstimation();
    
    // Initialize 3D points from stereo images
    void initialize3D(const cv::Mat& left_image, 
                     const cv::Mat& right_image,
                     std::vector<cv::Point3f>& points3d,
                     std::vector<cv::Mat>&  points_3d_descriptors,  // Store descriptors for valid points
                    std::vector<size_t>&  points_3d_valid_indices, 
                     std::vector<cv::KeyPoint>& left_keypoints,
                     std::vector<cv::KeyPoint>& right_keypoints,
                     std::vector<cv::DMatch>& good_matches,
                    bool verbose);
                        
    // PnP pose estimation
    void PnP(
        const cv::Mat& left_image,
        std::vector<cv::Point3f>& points3d,
        std::vector<cv::Mat>& points_3d_descriptors,
        std::vector<size_t>& points_3d_valid_indices,
        bool verbose
    );
    void getCurrentPose(cv::Mat& R_out, cv::Mat& t_out) {
        R_out = this->R.clone();
        t_out = this->t.clone();
    }


private:
    // Feature detectors and descriptors
    cv::Ptr<cv::FastFeatureDetector> fast_detector;
    cv::Ptr<cv::ORB> descriptor_extractor;
    cv::Ptr<cv::BFMatcher> matcher;
    cv::Ptr<cv::BFMatcher> knn_matcher;

    // Camera parameters
    const double focal_length = 718.856;      // KITTI camera parameters
    const cv::Point2d principal_point{607.1928, 185.2157};
    const double baseline = 0.54;             // 54cm baseline for KITTI

    // Current pose of right camera (world->camera)
    cv::Mat R;  // rotation matrix
    cv::Mat t;  // translation vector

    // Vector to store RANSAC inliers
    std::vector<int> inliers_;
};

