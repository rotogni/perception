#include "visualization.hpp"
#include <vector>
#include <iomanip>
#include <sstream>

Visualization::Visualization(const std::string& window_name)
    : window_name(window_name),
      viz_window(window_name),
      trajectory_window("Trajectory"),
      is_initialized(false) {
}

void Visualization::initializeWindows() {
    if (!is_initialized) {
        viz_window.setBackgroundColor(cv::viz::Color::white());
        viz_window.showWidget("Coordinate System", cv::viz::WCoordinateSystem(0.5));
        
        // Set initial camera pose for good trajectory view
        cv::Affine3d cam_pose;
        cv::Vec3d cam_pos(0.0, -1000.0, 150.0);
        cv::Vec3d cam_focal(0.0, 0.0, 150.0);
        cv::Vec3d cam_up(0.0, 0.0, -1.0);
        cam_pose = cv::viz::makeCameraPose(cam_pos, cam_focal, cam_up);
        viz_window.setViewerPose(cam_pose);
        
        is_initialized = true;
    }
}

void Visualization::add2DLegend(cv::Mat& display_image, size_t num_points, size_t num_poses, 
                               float min_depth, float max_depth) {
    // Create legend overlay on the image
    int legend_x = 100;
    int legend_y = 0;
    int line_height = 250;

    
    // Trajectory legend
    legend_y += line_height;
    cv::line(display_image, cv::Point(legend_x, legend_y), cv::Point(legend_x + 300, legend_y), 
             cv::Scalar(0, 255, 0), 30); // Green line
    cv::putText(display_image, "Ground Truth", cv::Point(legend_x + 400, legend_y + 50), 
                cv::FONT_HERSHEY_SIMPLEX, 4, cv::Scalar(0, 255, 0), 10);
    
    legend_y += line_height;
    cv::line(display_image, cv::Point(legend_x, legend_y), cv::Point(legend_x + 300, legend_y), 
             cv::Scalar(0, 0, 255), 30); // Red line
    cv::putText(display_image, "Estimated", cv::Point(legend_x + 400, legend_y + 50), 
                cv::FONT_HERSHEY_SIMPLEX, 4, cv::Scalar(0, 0, 255), 10);
    
    // Point cloud legend
    legend_y += line_height;
    cv::circle(display_image, cv::Point(legend_x + 200, legend_y), 12, cv::Scalar(255, 0, 0), -1); // Blue
    cv::putText(display_image, "Landmarks", cv::Point(legend_x + 400, legend_y + 50), 
                cv::FONT_HERSHEY_SIMPLEX, 4, cv::Scalar(255, 0, 0), 10);
    
}

void Visualization::showGroundTruthTrajectory(const std::vector<cv::Point3f>& gt_trajectory) {
    if (!gt_trajectory.empty()) {
        cv::viz::WPolyLine gt_poly(gt_trajectory, cv::viz::Color::green());
        gt_poly.setRenderingProperty(cv::viz::LINE_WIDTH, 3.0);
        viz_window.showWidget("GT_Trajectory", gt_poly);
    }
}

void Visualization::showStereoMatches(const cv::Mat& left_image,
                                    const cv::Mat& right_image,
                                    const std::vector<cv::KeyPoint>& left_keypoints,
                                    const std::vector<cv::KeyPoint>& right_keypoints,
                                    const std::vector<cv::DMatch>& matches) {
    // Only create visualization if we have matches
    if (matches.empty()) {
        return;
    }

    cv::Mat img_matches;
    cv::drawMatches(left_image, left_keypoints,
                    right_image, right_keypoints,
                    matches, img_matches,
                    cv::Scalar::all(-1),
                    cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::DEFAULT);

    cv::namedWindow("Stereo Matches", cv::WINDOW_AUTOSIZE);
    cv::imshow("Stereo Matches", img_matches);
    cv::pollKey(); // Non-blocking key check
}

// Add a separate legend window
void Visualization::showLegendWindow(size_t num_points, size_t num_poses, 
                                   float min_depth, float max_depth) {
    // Create a dedicated legend image
    cv::Mat legend_image = cv::Mat::zeros(1000, 1500, CV_8UC3);
    legend_image.setTo(cv::Scalar(250, 250, 250)); // Light gray background
    
    add2DLegend(legend_image, num_points, num_poses, min_depth, max_depth);
    
    cv::namedWindow("Legend", cv::WINDOW_AUTOSIZE);
    cv::imshow("Legend", legend_image);
    cv::pollKey();
}

void Visualization::updatePointCloud(const std::vector<cv::Point3f>& points3d,
                                   const std::vector<cv::Point3f>& trajectory_points) {
    if (points3d.empty()) {
        return;
    }

    // Create point cloud visualization
    cv::Mat points_mat(1, points3d.size(), CV_32FC3);
    cv::Mat colors(1, points3d.size(), CV_8UC3);
    
    cv::Point3f* ptr = points_mat.ptr<cv::Point3f>(0);
    cv::Vec3b* color_ptr = colors.ptr<cv::Vec3b>(0);

    // Color points based on depth from current camera position
    cv::Point3f cam_pos(0,0,0);
    if (!trajectory_points.empty()) {
        cam_pos = trajectory_points.back();
    }
    
    // Compute depth statistics for coloring
    std::vector<float> depths(points3d.size());
    float min_depth = std::numeric_limits<float>::max();
    float max_depth = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < points3d.size(); i++) {
        depths[i] = cv::norm(points3d[i] - cam_pos);
        min_depth = std::min(min_depth, depths[i]);
        max_depth = std::max(max_depth, depths[i]);
    }
    
    // Color points by normalized depth (blue=close, red=far)
    for (size_t i = 0; i < points3d.size(); i++) {
        ptr[i] = points3d[i];
        float norm_depth = (max_depth > min_depth) ? (depths[i] - min_depth) / (max_depth - min_depth) : 0.0f;
        color_ptr[i] = cv::Vec3b(
            static_cast<uchar>(255),           // Red component (far)
            static_cast<uchar>(0),  // Green component 
            static_cast<uchar>(0)   // Blue component (close)
        );
    }

    // Update point cloud widget
    cv::viz::WCloud cloud(points_mat, colors);
    cloud.setRenderingProperty(cv::viz::POINT_SIZE, 3);
    cloud.setRenderingProperty(cv::viz::REPRESENTATION, cv::viz::REPRESENTATION_POINTS);
    viz_window.showWidget("Point Cloud", cloud);

    // Visualize estimated trajectory
    if (!trajectory_points.empty()) {
        cv::viz::WPolyLine trajectory_widget(trajectory_points, cv::viz::Color::red());
        trajectory_widget.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
        viz_window.showWidget("Trajectory", trajectory_widget);

        // Add trajectory points as spheres (reduced frequency)
        const size_t sphere_spacing = std::max(size_t(1), trajectory_points.size() / 50);
        for (size_t i = 0; i < trajectory_points.size(); i += sphere_spacing) {
            std::string sphere_name = "trajectory_point_" + std::to_string(i/sphere_spacing);
            cv::viz::WSphere sphere(trajectory_points[i], 0.8, 10, cv::viz::Color::red());
            viz_window.showWidget(sphere_name, sphere);
        }
    }
    
    // Show 2D legend window
    showLegendWindow(points3d.size(), trajectory_points.size(), min_depth, max_depth);
    
    viz_window.spinOnce(1);
}


void Visualization::clearPointCloud() {
    try {
        viz_window.removeWidget("Point Cloud");
        viz_window.removeWidget("Trajectory");
        viz_window.removeWidget("Dynamic_Stats");
        
        // Remove trajectory spheres
        for (size_t i = 0; i < 50; i++) {
            std::string sphere_name = "trajectory_point_" + std::to_string(i);
            try {
                viz_window.removeWidget(sphere_name);
            } catch (...) {
                break;
            }
        }
    } catch (...) {
        // Widgets didn't exist
    }
}

bool Visualization::isWindowClosed() const {
    return viz_window.wasStopped();
}

void Visualization::cleanup() {
    cv::destroyAllWindows();
}