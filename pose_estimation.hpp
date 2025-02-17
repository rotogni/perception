#ifndef POSE_ESTIMATION_HPP
#define POSE_ESTIMATION_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class PoseEstimation {
public:
    PoseEstimation();
    
    // Initialize 3D points from stereo images
    void initialize3D(const cv::Mat& left_image, 
                     const cv::Mat& right_image,
                     std::vector<cv::Point3f>& points3d,
                     std::vector<cv::KeyPoint>& left_keypoints,
                     std::vector<cv::KeyPoint>& right_keypoints,
                     std::vector<cv::DMatch>& good_matches);

    // Feature detection and description
    void detectAndCompute(const cv::Mat& image,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::Mat& descriptors);

    // Feature matching between reference frame and current frame
    void matchFeatures(const cv::Mat& reference_image,
                      const cv::Mat& current_image,
                      const std::vector<cv::KeyPoint>& reference_keypoints,
                      std::vector<cv::KeyPoint>& current_keypoints,
                      std::vector<cv::DMatch>& matches);

    // PnP pose estimation
    void PnP(const cv::Mat& right_image,
             std::vector<cv::Point3f>& points3d,
             std::vector<cv::KeyPoint>& left_keypoints,
             std::vector<cv::KeyPoint>& right_keypoints,
             std::vector<cv::DMatch>& good_matches);

private:
    // Feature detectors and descriptors
    cv::Ptr<cv::FastFeatureDetector> fast_detector;
    cv::Ptr<cv::ORB> descriptor_extractor;
    cv::Ptr<cv::BFMatcher> matcher;

    // Camera parameters
    const double focal_length = 718.856;      // KITTI camera parameters
    const cv::Point2d principal_point{607.1928, 185.2157};
    const double baseline = 0.54;             // 54cm baseline for KITTI

    // Current pose
    cv::Mat R;  // rotation matrix
    cv::Mat t;  // translation vector
};

#endif // POSE_ESTIMATION_HPP
