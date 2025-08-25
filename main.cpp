#include <sstream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <iomanip>
#include <fstream>
#include <map>
#include "types.hpp"
#include "feature_detector.hpp"
#include "pose_estimation.hpp"
#include "visualization.hpp"

// Helper function to convert Pose to Point3f for visualization
cv::Point3f poseToPoint3f(const Pose &pose)
{
    // Extract translation vector
    cv::Point3f position;
    position.x = pose.t.at<double>(0);
    position.y = pose.t.at<double>(1);
    position.z = pose.t.at<double>(2);
    return position;
}

int main(){

    // Load KITTI ground truth trajectory
    std::vector<cv::Point3f> gt_trajectory;
    std::ifstream gt_file("Datasets/kitti/poses/05.txt");
    if (gt_file.is_open()) {
        std::string line;
        while (std::getline(gt_file, line)) {
            std::istringstream iss(line);
            double m[12];
            for (int i = 0; i < 12; ++i) iss >> m[i];
            // Translation is last column (m[3], m[7], m[11])
            gt_trajectory.emplace_back(m[3], m[7], m[11]);
        }
        gt_file.close();
        std::cout << "Loaded KITTI ground truth poses: " << gt_trajectory.size() << std::endl;
    } else {
        std::cout << "Could not open ground truth file!" << std::endl;
    }
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    // INPUT PARAMETERS
    bool verbose = false;
    const int KEYFRAME_INTERVAL = 2;

    // Create objects
    PoseEstimation pose_estimation;
    Visualization visualizer("3D Scene");

    // Initialize visualization
    visualizer.initializeWindows();

    // Variables to store reference frame data
    cv::Mat reference_image;
    std::vector<cv::Point3f> reference_points3d;
    std::vector<cv::KeyPoint> reference_left_kps, reference_right_kps;
    std::vector<cv::DMatch> reference_matches;

    // Initialize vectors for current frame
    std::vector<cv::Point3f> points3d;
    std::vector<cv::Mat> points_3d_descriptors;
    std::vector<size_t> points_3d_valid_indices;
    std::vector<cv::KeyPoint> left_keypoints, right_keypoints;
    std::vector<cv::DMatch> matches;

    // Store trajectory of camera
    std::vector<Pose> trajectory;
    std::vector<cv::Point3f> trajectory_points;
    

    // Load initial stereo image pair
    std::stringstream ss_left, ss_right;
    ss_left << "Datasets/kitti/05/image_0/"
            << std::setfill('0') << std::setw(6) << 0 << ".png";
    ss_right << "Datasets/kitti/05/image_1/"
             << std::setfill('0') << std::setw(6) << 0 << ".png";

    cv::Mat left_image = cv::imread(ss_left.str());
    cv::Mat right_image = cv::imread(ss_right.str());

    if (left_image.empty() || right_image.empty())
    {
        std::cout << "Error: Could not load initial images" << std::endl;
        return -1;
    }

    // Initialize 3D points
    pose_estimation.initialize3D(left_image, right_image, points3d, points_3d_descriptors, points_3d_valid_indices,
                                 left_keypoints, right_keypoints, matches, verbose);

    reference_image = left_image.clone();
    reference_points3d = points3d;

    if (verbose)
    {
        std::cout << "Initialized with " << points3d.size() << " 3D points" << std::endl;
        reference_left_kps = left_keypoints;
        reference_right_kps = right_keypoints;
    }

    // Main processing loop
    for (int i = 1; i < 5000; i++)
    {
        // Load stereo image pair
        ss_left.str(""); ss_left.clear();
        ss_right.str(""); ss_right.clear();
        ss_left << "Datasets/kitti/05/image_0/"
                << std::setfill('0') << std::setw(6) << i << ".png";
        ss_right << "Datasets/kitti/05/image_1/"
                 << std::setfill('0') << std::setw(6) << i << ".png";

        left_image = cv::imread(ss_left.str());
        right_image = cv::imread(ss_right.str());

        if (left_image.empty() || right_image.empty())
        {
            std::cout << "Error: Could not load images for frame " << i << std::endl;
            continue;
        }

        // Process keyframes
        if (i % KEYFRAME_INTERVAL == 0)
        {
            // Calculate current pose using PnP
            pose_estimation.PnP(left_image, points3d, points_3d_descriptors, points_3d_valid_indices, verbose);

            // Get current pose and store in trajectory
            Pose current_pose;
            pose_estimation.getCurrentPose(current_pose.R, current_pose.t);
            trajectory.push_back(current_pose);

            // Calculate new 3D points
            pose_estimation.initialize3D(left_image, right_image, points3d, points_3d_descriptors, points_3d_valid_indices,
                                         left_keypoints, right_keypoints, matches, verbose);

            reference_image = left_image.clone();
            reference_points3d = points3d;

            if (verbose)
            {
                std::cout << "Frame " << i << ": New reference frame with "
                          << points3d.size() << " 3D points" << std::endl;
                reference_left_kps = left_keypoints;
                reference_right_kps = right_keypoints;
            }
        }
        else
        {
            // Run PnP for non-keyframes
            pose_estimation.PnP(left_image, points3d, points_3d_descriptors, points_3d_valid_indices, verbose);

            Pose current_pose;
            pose_estimation.getCurrentPose(current_pose.R, current_pose.t);
            trajectory.push_back(current_pose);
            trajectory_points.push_back(poseToPoint3f(current_pose));

            if (verbose)
            {
                std::cout << "Frame " << i << " pose:" << std::endl;
                std::cout << "R = " << std::endl << current_pose.R << std::endl;
                std::cout << "t = " << current_pose.t.t() << std::endl;
            }
        }

        // Show visualizations
        visualizer.showGroundTruthTrajectory(gt_trajectory);
        visualizer.showStereoMatches(left_image, right_image,
                                     left_keypoints, right_keypoints,
                                     matches);
        visualizer.updatePointCloud(points3d, trajectory_points);

        // Wait for first frame
        if (i == 1) {
            cv::waitKey(0);
        }
    }
    
    visualizer.cleanup();
    return 0;
}