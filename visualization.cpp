#include "visualization.hpp"

Visualization::Visualization(const std::string& window_name)
    : window_name(window_name),
      viz_window(window_name),
      trajectory_window("Trajectory"),
      is_initialized(false) {
}

void Visualization::initializeWindows() {
    if (!is_initialized) {
        viz_window.setBackgroundColor(cv::viz::Color::white());
        viz_window.showWidget("Coordinate System", cv::viz::WCoordinateSystem());
        is_initialized = true;
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

    // Find depth range
    float min_z = std::numeric_limits<float>::max();
    float max_z = std::numeric_limits<float>::lowest();
    
    #pragma omp parallel for reduction(min:min_z) reduction(max:max_z)
    for (size_t i = 0; i < points3d.size(); i++) {
        min_z = std::min(min_z, points3d[i].z);
        max_z = std::max(max_z, points3d[i].z);
    }

    // Color points based on depth
    #pragma omp parallel for
    for (size_t i = 0; i < points3d.size(); i++) {
        ptr[i] = points3d[i];
        float normalized_z = (points3d[i].z - min_z) / (max_z - min_z);
        color_ptr[i] = cv::Vec3b(
            static_cast<uchar>(255 * (1.0f - normalized_z)),
            static_cast<uchar>(255 * normalized_z),
            0);
    }

    // Update point cloud widget
    cv::viz::WCloud cloud(points_mat, colors);
    cloud.setRenderingProperty(cv::viz::POINT_SIZE, 3);
    cloud.setRenderingProperty(cv::viz::REPRESENTATION, cv::viz::REPRESENTATION_POINTS);
    viz_window.showWidget("Point Cloud", cloud);

    // Visualize trajectory if available
    if (!trajectory_points.empty()) {
        // Create trajectory line
        cv::viz::WPolyLine trajectory_widget(trajectory_points, cv::viz::Color::red());
        trajectory_widget.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
        viz_window.showWidget("Trajectory", trajectory_widget);

        // Add trajectory points as spheres (with reduced frequency)
        const size_t sphere_spacing = std::max(size_t(1), trajectory_points.size() / 50);
        for (size_t i = 0; i < trajectory_points.size(); i += sphere_spacing) {
            std::string sphere_name = "trajectory_point_" + std::to_string(i/sphere_spacing);
            cv::viz::WSphere sphere(trajectory_points[i], 0.05, 10, cv::viz::Color::red());
            viz_window.showWidget(sphere_name, sphere);
        }
    }

    viz_window.spinOnce(1);
}

void Visualization::clearPointCloud() {
    try {
        viz_window.removeWidget("Point Cloud");
        viz_window.removeWidget("Trajectory");
        
        // Remove any existing trajectory spheres
        for (size_t i = 0; i < 50; i++) {  // Maximum number of spheres we could have created
            std::string sphere_name = "trajectory_point_" + std::to_string(i);
            try {
                viz_window.removeWidget(sphere_name);
            } catch (...) {
                break;
            }
        }
    } catch (...) {
        // Widget didn't exist, which is fine
    }
}

bool Visualization::isWindowClosed() const {
    return viz_window.wasStopped();
}

void Visualization::cleanup() {
    cv::destroyAllWindows();
}