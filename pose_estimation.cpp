#include "pose_estimation.hpp"
#include <iostream>

PoseEstimation::PoseEstimation() {
    // Initialize feature detector
    fast_detector = cv::FastFeatureDetector::create();
    fast_detector->setThreshold(40);
    fast_detector->setNonmaxSuppression(true);
    
    // Initialize ORB descriptor for feature matching
    descriptor_extractor = cv::ORB::create(
        2000,               // nfeatures
        1.2f,              // scaleFactor
        8,                 // nlevels
        31,                // edgeThreshold
        0,                 // firstLevel
        2,                 // WTA_K
        cv::ORB::HARRIS_SCORE, // scoreType
        31,                // patchSize
        20                 // fastThreshold
    );
    
    // Initialize matcher with Hamming distance (for binary descriptors like ORB)
    matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
}

void PoseEstimation::initialize3D(const cv::Mat& left_image,
                                const cv::Mat& right_image,
                                std::vector<cv::Point3f>& points3d,
                                std::vector<cv::KeyPoint>& left_keypoints,
                                std::vector<cv::KeyPoint>& right_keypoints,
                                std::vector<cv::DMatch>& good_matches) {
    // Detect features in both images
    fast_detector->detect(left_image, left_keypoints);
    fast_detector->detect(right_image, right_keypoints);
    
    // Compute descriptors
    cv::Mat left_descriptors, right_descriptors;
    descriptor_extractor->compute(left_image, left_keypoints, left_descriptors);
    descriptor_extractor->compute(right_image, right_keypoints, right_descriptors);
    
    // Match features
    std::vector<cv::DMatch> matches;
    if (!left_descriptors.empty() && !right_descriptors.empty()) {
        matcher->match(left_descriptors, right_descriptors, matches);
    }
    
    // Filter matches using ratio test and epipolar constraint
    const float ratio_thresh = 0.75f;
    const float max_y_diff = 2.0f;  // Maximum vertical disparity
    good_matches.clear();
    std::vector<cv::Point2f> left_points, right_points;
    
    for (const auto& match : matches) {
        cv::Point2f left_pt = left_keypoints[match.queryIdx].pt;
        cv::Point2f right_pt = right_keypoints[match.trainIdx].pt;
        
        // Check vertical disparity
        if (std::abs(left_pt.y - right_pt.y) > max_y_diff) {
            continue;
        }
        
        // Check horizontal disparity (should be positive for stereo)
        float disparity = left_pt.x - right_pt.x;
        if (disparity <= 0) {
            continue;
        }
        
        if (match.distance < ratio_thresh * 50) {
            good_matches.push_back(match);
            left_points.push_back(left_pt);
            right_points.push_back(right_pt);
        }
    }
    
    if (good_matches.size() >= 8) {
        // Create camera matrix K
        cv::Mat K = (cv::Mat_<double>(3,3) <<
            focal_length, 0, principal_point.x,
            0, focal_length, principal_point.y,
            0, 0, 1);
        
        // For stereo, we know R is identity and t is [baseline, 0, 0]
        cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat t = (cv::Mat_<double>(3,1) << -baseline, 0, 0);  // Negative baseline for right camera
        
        // First camera matrix [I|0]
        cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
        cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
        I.copyTo(P1(cv::Rect(0, 0, 3, 3)));
        P1 = K * P1;
        
        // Second camera matrix [R|t]
        cv::Mat P2;
        cv::hconcat(R, t, P2);
        P2 = K * P2;
        
        // Triangulate points
        cv::Mat points_4d;
        cv::triangulatePoints(P1, P2, left_points, right_points, points_4d);
        
        // Convert homogeneous coordinates to 3D points with filtering
        points3d.clear();
        for (int i = 0; i < points_4d.cols; i++) {
            double w = points_4d.at<float>(3, i);
            if (std::abs(w) > 1e-10) {  // Check for valid homogeneous coordinate
                cv::Point3f p(
                    points_4d.at<float>(0, i) / w,
                    points_4d.at<float>(1, i) / w,
                    points_4d.at<float>(2, i) / w
                );
                
                // Basic depth check
                if (p.z > 0 && p.z < 50.0) {  // Points should be in front and within reasonable distance
                    points3d.push_back(p);
                }
            }
        }
        
        std::cout << "Triangulation stats:" << std::endl;
        std::cout << "Matches used: " << good_matches.size() << std::endl;
        std::cout << "Points triangulated: " << points3d.size() << std::endl;
    }
}

void PoseEstimation::PnP(const cv::Mat& right_image,
    std::vector<cv::Point3f>& points3d,
    std::vector<cv::KeyPoint>& reference_keypoints,
    std::vector<cv::KeyPoint>& current_keypoints,
    std::vector<cv::DMatch>& good_matches) {
    
    std::cout << "PnP Debug Info:" << std::endl;
    std::cout << "Total 3D points available: " << points3d.size() << std::endl;
    std::cout << "Reference keypoints: " << reference_keypoints.size() << std::endl;
    std::cout << "Current keypoints: " << current_keypoints.size() << std::endl;
    std::cout << "Good matches: " << good_matches.size() << std::endl;

    // Convert keypoints to Point2f for PnP algorithm
    std::vector<cv::Point2f> points2d;
    std::vector<cv::Point3f> matched_3d_points;
    std::vector<cv::DMatch> valid_matches;

    // Only use points that have valid matches and are within the points3d range
    for (const auto& match : good_matches) {
        if (match.queryIdx < points3d.size() && match.trainIdx < current_keypoints.size()) {
            matched_3d_points.push_back(points3d[match.queryIdx]);
            points2d.push_back(current_keypoints[match.trainIdx].pt);
            valid_matches.push_back(match);
        }
    }

    std::cout << "After filtering:" << std::endl;
    std::cout << "Matched 3D points: " << matched_3d_points.size() << std::endl;
    std::cout << "2D points: " << points2d.size() << std::endl;

    // Make sure we have enough matches for PnP
    if (matched_3d_points.size() < 4 || points2d.size() < 4) {
        std::cout << "Not enough points for PnP. Required: 4, Got: " 
                  << matched_3d_points.size() << std::endl;
        return;
    }

    // Verify point vectors have same size
    if (matched_3d_points.size() != points2d.size()) {
        std::cout << "Error: Point count mismatch. 3D points: " 
                  << matched_3d_points.size() << ", 2D points: " 
                  << points2d.size() << std::endl;
        return;
    }

    // Create camera matrix K
    cv::Mat K = (cv::Mat_<double>(3,3) <<
        focal_length, 0, principal_point.x,
        0, focal_length, principal_point.y,
        0, 0, 1);

    // Distortion coefficients (assume zero if not calibrated)
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

    // Output rotation and translation
    cv::Mat rvec, tvec;
    bool success = false;

    try {
        // Solve PnP using RANSAC for robustness
        success = cv::solvePnPRansac(matched_3d_points, points2d, K, distCoeffs, rvec, tvec,
            false,          // useExtrinsicGuess
            100,           // iterationsCount
            8.0,           // reprojectionError
            0.99,          // confidence
            cv::noArray(), // inliers
            cv::SOLVEPNP_ITERATIVE);

        if (success) {
            // Update the matches to only include valid ones
            good_matches = valid_matches;

            // Convert rotation vector to rotation matrix
            cv::Mat R;
            cv::Rodrigues(rvec, R);

            // Store or use the pose
            this->R = R;
            this->t = tvec;

            // Optional: Calculate reprojection error
            std::vector<cv::Point2f> reprojected_points;
            cv::projectPoints(matched_3d_points, rvec, tvec, K, distCoeffs, reprojected_points);
            
            double total_error = 0;
            for (size_t i = 0; i < points2d.size(); i++) {
                double error = cv::norm(points2d[i] - reprojected_points[i]);
                total_error += error;
            }
            double avg_error = total_error / points2d.size();
            std::cout << "PnP successful! Average reprojection error: " 
                      << avg_error << " pixels" << std::endl;
        } else {
            std::cout << "PnP failed to find a solution" << std::endl;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
    }
}

void PoseEstimation::detectAndCompute(const cv::Mat& image,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors) {
// Use combined detectAndCompute for better efficiency
descriptor_extractor->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}


void PoseEstimation::matchFeatures(const cv::Mat& reference_image,
    const cv::Mat& current_image,
    const std::vector<cv::KeyPoint>& reference_keypoints,
    std::vector<cv::KeyPoint>& current_keypoints,
    std::vector<cv::DMatch>& matches) {

std::cout << "Feature Matching Debug Info:" << std::endl;
std::cout << "Reference keypoints: " << reference_keypoints.size() << std::endl;

// Detect and compute for both frames
cv::Mat reference_descriptors, current_descriptors;
std::vector<cv::KeyPoint> temp_ref_keypoints;

// For reference frame
descriptor_extractor->detectAndCompute(reference_image, cv::noArray(), 
            temp_ref_keypoints, reference_descriptors);

// For current frame
descriptor_extractor->detectAndCompute(current_image, cv::noArray(), 
            current_keypoints, current_descriptors);

std::cout << "Current keypoints detected: " << current_keypoints.size() << std::endl;

// Match features
matches.clear();

if (!reference_descriptors.empty() && !current_descriptors.empty()) {
// Use simple matching
matcher->match(reference_descriptors, current_descriptors, matches);

std::cout << "Initial matches found: " << matches.size() << std::endl;

// Filter matches based on distance
std::vector<cv::DMatch> good_matches;
double max_dist = 0;
double min_dist = std::numeric_limits<double>::max();

// Calculate min and max distances
for (const auto& match : matches) {
double dist = match.distance;
if (dist < min_dist) min_dist = dist;
if (dist > max_dist) max_dist = dist;
}

std::cout << "Distance range -- Min: " << min_dist << " Max: " << max_dist << std::endl;

// Keep only good matches (those whose distance is less than 2*min_dist)
const double dist_thresh = std::min(2 * min_dist, 75.0);
for (const auto& match : matches) {
if (match.distance <= dist_thresh) {
// Remap match indices to use original reference keypoints
cv::DMatch adjusted_match = match;
adjusted_match.queryIdx = match.queryIdx;  // This should be within bounds of reference_keypoints
if (adjusted_match.queryIdx < reference_keypoints.size() && 
adjusted_match.trainIdx < current_keypoints.size()) {
good_matches.push_back(adjusted_match);
}
}
}

matches = good_matches;
std::cout << "Good matches after distance filtering: " << matches.size() << std::endl;
} else {
std::cout << "Warning: Cannot perform matching!" << std::endl;
std::cout << "Reference descriptors empty: " << reference_descriptors.empty() << std::endl;
std::cout << "Current descriptors empty: " << current_descriptors.empty() << std::endl;
}
}