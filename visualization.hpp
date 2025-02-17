#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

class Visualization {
public:
    Visualization(const std::string& window_name = "3D Point Cloud");
    void initializeWindows();
    void showStereoMatches(const cv::Mat& left_image,
                          const cv::Mat& right_image,
                          const std::vector<cv::KeyPoint>& left_keypoints,
                          const std::vector<cv::KeyPoint>& right_keypoints,
                          const std::vector<cv::DMatch>& matches);
    void updatePointCloud(const std::vector<cv::Point3f>& points3d);
    void clearPointCloud();
    bool isWindowClosed() const;
    void cleanup();

private:
    cv::viz::Viz3d viz_window;
    std::string window_name;
    bool is_initialized;
};

#endif 