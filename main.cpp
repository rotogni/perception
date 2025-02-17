#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <iomanip>
#include "feature_detector.hpp"
#include "pose_estimation.hpp"
#include "visualization.hpp"

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    
    // Create objects
    PoseEstimation pose_estimation;
    Visualization visualizer("3D Point Cloud");
    
    // Initialize visualization
    visualizer.initializeWindows();
    
    // Variables to store reference frame data
    cv::Mat reference_image;  // Store reference image
    std::vector<cv::Point3f> reference_points3d;
    std::vector<cv::KeyPoint> reference_left_kps, reference_right_kps;
    std::vector<cv::DMatch> reference_matches;
    
    // Main loop
    for (int i = 0; i < 50; i++) {
        // Load stereo image pair
        std::stringstream ss_left, ss_right;
        ss_left << "Datasets/kitti/05/image_0/"
                << std::setfill('0') << std::setw(6) << i << ".png";
        ss_right << "Datasets/kitti/05/image_1/"
                << std::setfill('0') << std::setw(6) << i << ".png";
        
        cv::Mat left_image = cv::imread(ss_left.str());
        cv::Mat right_image = cv::imread(ss_right.str());
        
        if(left_image.empty() || right_image.empty()) {
            std::cout << "Error: Could not load images" << std::endl;
            continue;
        }
        
        // Initialize vectors for current frame
        std::vector<cv::Point3f> points3d;
        std::vector<cv::KeyPoint> left_keypoints, right_keypoints;
        std::vector<cv::DMatch> good_matches;
        
        // Every 5 frames, perform new 3D reconstruction
        if (i % 5 == 0) {
            pose_estimation.initialize3D(left_image, right_image, points3d,
                                      left_keypoints, right_keypoints, good_matches);
            
            reference_image = right_image.clone();  // Store reference image
            reference_points3d = points3d;
            reference_left_kps = left_keypoints;
            reference_right_kps = right_keypoints;
            reference_matches = good_matches;
            
            std::cout << "Frame " << i << ": New reference frame with "
                        << points3d.size() << " 3D points" << std::endl;
        }
        // For the next 5 frames, use PnP
        else {
            // Extract features and match with reference frame
            std::vector<cv::KeyPoint> current_keypoints;
            std::vector<cv::DMatch> pnp_matches;
            
            pose_estimation.matchFeatures(reference_image, right_image,
                                        reference_right_kps, current_keypoints,
                                        pnp_matches);
            
            // Perform PnP
            pose_estimation.PnP(right_image, reference_points3d,
                              reference_right_kps, current_keypoints, pnp_matches);
            
            // Update points3d and matches for visualization
            points3d = reference_points3d;
            good_matches = pnp_matches;
            left_keypoints = reference_right_kps;
            right_keypoints = current_keypoints;
            
            std::cout << "Frame " << i << ": PnP estimation with "
                     << pnp_matches.size() << " matches" << std::endl;
        }
        
        // Show visualizations
        visualizer.showStereoMatches(left_image, right_image,
                                   left_keypoints, right_keypoints,
                                   good_matches);
        visualizer.updatePointCloud(points3d);
        
        // Wait for key press to continue
        char key = cv::waitKey(0);
        if (key == 'q' || key == 'Q' || visualizer.isWindowClosed()) {
            break;
        }
    }
    
    visualizer.cleanup();
    return 0;
}