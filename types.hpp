#pragma once
#include <opencv2/opencv.hpp>

struct Pose {
    cv::Mat R;  // 3x3 rotation matrix
    cv::Mat t;  // 3x1 translation vector
};