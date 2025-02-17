// visualization.cpp
#include "visualization.hpp"

Visualization::Visualization(const std::string& window_name)
    : window_name(window_name),
    viz_window(window_name),
    is_initialized(false) {
}

void Visualization::initializeWindows() {
    if (!is_initialized) {
        // Set up 3D visualization window
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
    cv::waitKey(1);  // Add a small delay to ensure window updates
}

void Visualization::clearPointCloud() {
    try {
        viz_window.removeWidget("Point Cloud");
    } catch (...) {
        // Widget didn't exist, which is fine
    }
}

void Visualization::updatePointCloud(const std::vector<cv::Point3f>& points3d) {
    if (points3d.empty()) {
        std::cout << "Warning: Empty point cloud, skipping visualization" << std::endl;
        return;
    }

    try {
        clearPointCloud();

        std::vector<cv::Point3f> valid_points;
        valid_points.reserve(points3d.size());
        
        for (const auto& pt : points3d) {
            if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z) &&
                std::abs(pt.z) < 100.0) {
                valid_points.push_back(pt);
            }
        }

        std::cout << "Valid points: " << valid_points.size() << " out of " << points3d.size() << std::endl;

        if (valid_points.empty()) {
            std::cout << "Warning: No valid points after filtering" << std::endl;
            return;
        }

        cv::Mat points_mat(1, valid_points.size(), CV_32FC3);
        cv::Point3f* ptr = points_mat.ptr<cv::Point3f>(0);
        
        cv::Mat colors(1, valid_points.size(), CV_8UC3);
        cv::Vec3b* color_ptr = colors.ptr<cv::Vec3b>(0);

        float min_z = std::numeric_limits<float>::max();
        float max_z = std::numeric_limits<float>::lowest();
        
        for (const auto& pt : valid_points) {
            min_z = std::min(min_z, pt.z);
            max_z = std::max(max_z, pt.z);
        }

        for (size_t i = 0; i < valid_points.size(); i++) {
            ptr[i] = valid_points[i];
            
            float normalized_z = (valid_points[i].z - min_z) / (max_z - min_z);
            color_ptr[i] = cv::Vec3b(
                static_cast<uchar>(255 * (1.0f - normalized_z)),
                static_cast<uchar>(255 * normalized_z),
                0
            );
        }

        cv::viz::WCloud cloud(points_mat, colors);
        cloud.setRenderingProperty(cv::viz::POINT_SIZE, 3);

        viz_window.showWidget("Point Cloud", cloud);
        viz_window.spinOnce(1);

        std::cout << "Point cloud updated successfully. Depth range: " 
                  << min_z << " to " << max_z << std::endl;

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error in updatePointCloud: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in updatePointCloud: " << e.what() << std::endl;
    }
}

bool Visualization::isWindowClosed() const {
    return viz_window.wasStopped();
}

void Visualization::cleanup() {
    cv::destroyAllWindows();
}