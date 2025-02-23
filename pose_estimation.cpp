#include "pose_estimation.hpp"
#include <iostream>

PoseEstimation::PoseEstimation() {
    // Initialize feature detector
    fast_detector = cv::FastFeatureDetector::create();
    fast_detector->setThreshold(40);
    fast_detector->setNonmaxSuppression(true);


    // Initialize pose as identity
    R = cv::Mat::eye(3, 3, CV_64F);
    t = cv::Mat::zeros(3, 1, CV_64F);
    
    // Initialize ORB descriptor for feature matching
    descriptor_extractor = cv::ORB::create(
        500,     // nfeatures - Increased for better coverage
        1.3f,     // scaleFactor - Slightly increased for better scale handling
        12,       // nlevels - More pyramid levels for scale invariance
        35,       // edgeThreshold - Increased to detect features closer to edges
        0,        // firstLevel - Keep default
        4,        // WTA_K - Increased for more robust descriptors
        cv::ORB::HARRIS_SCORE,  // scoreType - Harris corners are more stable
        32,       // patchSize - Adjusted for balance
        25        // fastThreshold - Lowered to detect more features
    );
    
    // Initialize matcher with Hamming distance (for binary descriptors like ORB)
    matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    // Create matcher without cross-check for kNN
    knn_matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    
}

void PoseEstimation::initialize3D(const cv::Mat& left_image,
                                const cv::Mat& right_image,
                                std::vector<cv::Point3f>& points3d,
                                std::vector<cv::Mat>&  points_3d_descriptors,  
                                std::vector<size_t>&  points_3d_valid_indices, 
                                std::vector<cv::KeyPoint>& left_keypoints,
                                std::vector<cv::KeyPoint>& right_keypoints,
                                std::vector<cv::DMatch>& matches,
                            bool verbose) {
    // Detect features in both images
    fast_detector->detect(left_image, left_keypoints);
    fast_detector->detect(right_image, right_keypoints);
    
    // Compute descriptors
    cv::Mat left_descriptors, right_descriptors;
    descriptor_extractor->compute(left_image, left_keypoints, left_descriptors);
    descriptor_extractor->compute(right_image, right_keypoints, right_descriptors);
    
    // Match features using kNN for Lowe's ratio test
    if (!left_descriptors.empty() && !right_descriptors.empty()) {
        std::vector<std::vector<cv::DMatch>> knn_matches;
        knn_matcher->knnMatch(left_descriptors, right_descriptors, knn_matches, 2);
        // Apply Lowe's ratio test with additional epipolar constraint
        const float ratio_thresh = 0.75f;
        const float max_y_diff = 2.0f;  // Maximum vertical disparity for stereo
        matches.clear();
        matches.reserve(knn_matches.size());
        
        for (const auto& match_pair : knn_matches) {
            if (match_pair.size() < 2) continue;
            
            const auto& m = match_pair[0];  // Best match
            const auto& n = match_pair[1];  // Second best match
            
            // Apply ratio test
            if (m.distance < ratio_thresh * n.distance) {
                // Apply epipolar constraint for stereo
                float y_diff = std::abs(left_keypoints[m.queryIdx].pt.y - 
                                      right_keypoints[m.trainIdx].pt.y);
                if (y_diff < max_y_diff) {
                    matches.push_back(m);
                }
            }
        }
        
        // Sort matches by distance
        std::sort(matches.begin(), matches.end(),
                 [](const cv::DMatch& a, const cv::DMatch& b) {
                     return a.distance < b.distance;
                 });
    }
    
    // Convert matches to point correspondences
    std::vector<cv::Point2f> left_points, right_points;
    cv::Mat matched_desc_left, matched_desc_right;
    left_points.reserve(matches.size());
    right_points.reserve(matches.size());

    // First collect the points
    for (const auto& m : matches) {
        left_points.push_back(left_keypoints[m.queryIdx].pt);
        right_points.push_back(right_keypoints[m.trainIdx].pt);
    }

    // Create descriptor matrices for matched points
    matched_desc_left = cv::Mat(matches.size(), left_descriptors.cols, left_descriptors.type());
    matched_desc_right = cv::Mat(matches.size(), right_descriptors.cols, right_descriptors.type());

    for (size_t i = 0; i < matches.size(); ++i) {
        left_descriptors.row(matches[i].queryIdx).copyTo(matched_desc_left.row(i));
        right_descriptors.row(matches[i].trainIdx).copyTo(matched_desc_right.row(i));
    }
    
    if (matches.size() >= 8) {

        // Get current pose (world->right_camera)

        // Create camera matrix K
        cv::Mat K = (cv::Mat_<double>(3,3) <<
            focal_length, 0, principal_point.x,
            0, focal_length, principal_point.y,
            0, 0, 1);

        
        // Left camera is reference at [I|0]
        cv::Mat P_left = cv::Mat::zeros(3, 4, CV_64F);
        cv::Mat eye = cv::Mat::eye(3, 3, CV_64F);
        eye.copyTo(P_left(cv::Rect(0, 0, 3, 3)));

        // Right camera offset by baseline [I|baseline]
        cv::Mat P_right = cv::Mat::zeros(3, 4, CV_64F);
        eye.copyTo(P_right(cv::Rect(0, 0, 3, 3)));
        cv::Mat baseline_vec = (cv::Mat_<double>(3,1) << -baseline, 0, 0);
        baseline_vec.copyTo(P_right(cv::Rect(3, 0, 1, 3)));

        // Apply camera matrix after
        P_left = K * P_left;
        P_right = K * P_right;
                
        // Triangulate points
        cv::Mat points_4d;
        cv::triangulatePoints(P_left, P_right, left_points,right_points, points_4d);
        
        // Convert homogeneous coordinates to 3D points with filtering
        // At the start of initialize3D
        points3d.clear();
        points_3d_descriptors.clear();
        points_3d_valid_indices.clear();
        points_3d_descriptors.reserve(points_4d.cols);
        points_3d_valid_indices.reserve(points_4d.cols);

        Pose current_pose;
        getCurrentPose(current_pose.R, current_pose.t); 

        for (int i = 0; i < points_4d.cols; i++) {
            double w = points_4d.at<float>(3, i);
            if (std::abs(w) > 1e-10) { // Check for valid homogeneous coordinate
                // Convert from homogeneous to 3D camera coordinates
                cv::Point3f p_cam(
                    points_4d.at<float>(0, i) / w,
                    points_4d.at<float>(1, i) / w,
                    points_4d.at<float>(2, i) / w
                );
                
                // Basic depth check in camera frame
                if (p_cam.z > 0 && p_cam.z < 50.0) {
                    // Transform to world frame
                    cv::Mat p_cam_mat = (cv::Mat_<double>(3,1) << p_cam.x, p_cam.y, p_cam.z);
                    cv::Mat p_world_mat = current_pose.R * p_cam_mat + current_pose.t;
                    
                    cv::Point3f p_world(
                        p_world_mat.at<double>(0,0),
                        p_world_mat.at<double>(1,0),
                        p_world_mat.at<double>(2,0)
                    );
                    
                    points3d.push_back(p_world);
                    points_3d_descriptors.push_back(matched_desc_left.row(i).clone());
                    points_3d_valid_indices.push_back(i);
                }
            }
        }

        // Create final descriptor matrix with only valid points
        cv::Mat landmark_desc;
        if (!points_3d_descriptors.empty()) {
            cv::vconcat(points_3d_descriptors, landmark_desc);
        }
        if(verbose){
        std::cout << "Triangulation stats:" << std::endl;
        std::cout << "Matches used: " << matches.size() << std::endl;
        std::cout << "Points triangulated: " << points3d.size() << std::endl;
        std::cout << "Descriptors stored: " << landmark_desc.rows << std::endl;
        std::cout << "Descriptors stored: " << points_3d_descriptors.size() << std::endl;
        }
    }
}

void PoseEstimation::PnP(const cv::Mat& left_image,
                        std::vector<cv::Point3f>& points3d,
                        std::vector<cv::Mat>& points_3d_descriptors,
                        std::vector<size_t>& points_3d_valid_indices,
                    bool verbose) {
    // Detect and compute features in current frame
    std::vector<cv::KeyPoint> current_keypoints;
    cv::Mat current_descriptors;
    // Detect features in left images
    fast_detector->detect(left_image, current_keypoints);
    // Compute descriptors
    descriptor_extractor->compute(left_image, current_keypoints, current_descriptors);

    cv::Mat landmark_desc;
        if (!points_3d_descriptors.empty()) {
            cv::vconcat(points_3d_descriptors, landmark_desc);
        }
    // Match using kNN for Lowe's ratio test
    std::vector<std::vector<cv::DMatch>> knn_matches;
    cv::Ptr<cv::BFMatcher> knn_matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    knn_matcher->knnMatch(landmark_desc, current_descriptors, knn_matches, 2);

    // Apply Lowe's ratio test
    std::vector<cv::DMatch> good_matches;
    const float ratio_thresh = 0.75f;
    good_matches.reserve(knn_matches.size());

    for (const auto& match_pair : knn_matches) {
        if (match_pair.size() < 2) continue;
        if (match_pair[0].distance < ratio_thresh * match_pair[1].distance) {
            good_matches.push_back(match_pair[0]);
        }
    }

    // Prepare data for PnP
    std::vector<cv::Point3f> matched_3d_points;
    std::vector<cv::Point2f> matched_2d_points;
    matched_3d_points.reserve(good_matches.size());
    matched_2d_points.reserve(good_matches.size());

    for (const auto& match : good_matches) {
        matched_3d_points.push_back(points3d[match.queryIdx]);
        matched_2d_points.push_back(current_keypoints[match.trainIdx].pt);
    }

    // Early return if not enough matches
    if (matched_3d_points.size() < 6) {  // Need at least 6 points for P3P + RANSAC
        if(verbose){
        std::cout << "not enough matches for PnP :" << matched_3d_points.size() << std::endl;
        }
        return;
    }

    // Create camera matrix K
    cv::Mat K = (cv::Mat_<double>(3,3) <<
    focal_length, 0, principal_point.x,
    0, focal_length, principal_point.y,
    0, 0, 1);

    // Initialize PnP parameters
    cv::Mat rvec, tvec;
    cv::Mat inlier_mask;
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);  // Assuming no distortion

    // Run PnP RANSAC
    bool success = cv::solvePnPRansac(
        matched_3d_points,
        matched_2d_points,
        K,
        dist_coeffs,
        rvec,
        tvec,
        false,      // useExtrinsicGuess
        100,        // iterationsCount
        8.0,        // reprojectionError threshold
        0.99,       // confidence
        inlier_mask,
        cv::SOLVEPNP_P3P
    );

    if (success) {
        // Convert rotation vector to matrix
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        // Calculate statistics
        int inlier_count = cv::countNonZero(inlier_mask);
        double points3d_percentage = 100.0 * matched_3d_points.size() / points3d.size();
        double inlier_percentage = 100.0 * inlier_count / matched_3d_points.size();
        double final_percentage = 100.0 * inlier_count / points3d.size();

        // Update class members with results
        this->R = R.t();  // Transpose of rotation matrix is its inverse
        this->t = -R.t() * tvec;  // Transform translation accordingly
        // Print statistics
        if(verbose){
        std::cout << "PnP RANSAC stats:" << std::endl;
        std::cout << "3D points used: " << points3d_percentage << "%" << std::endl;
        std::cout << "Inlier ratio: " << inlier_percentage << "%" << std::endl;
        std::cout << "Final percentage: " << final_percentage << "%" << std::endl;
        }
    }
    else {
        if(verbose){
        std::cout << "PnP RANSAC failed:" << std::endl;
        }
    }
}