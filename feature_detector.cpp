#include "feature_detector.hpp"

FeatureDetector::FeatureDetector(int threshold) : threshold(threshold) {
    fast_detector = cv::FastFeatureDetector::create();
    fast_detector->setThreshold(threshold);
    fast_detector->setNonmaxSuppression(true);
}

cv::Mat FeatureDetector::processImage(const cv::Mat& image, int& num_features) {
    // Detect FAST features
    std::vector<cv::KeyPoint> keypoints;
    fast_detector->detect(image, keypoints);
    
    // Draw keypoints
    cv::Mat image_with_keypoints;
    cv::drawKeypoints(image, keypoints, image_with_keypoints,
                     cv::Scalar(0, 255, 0),
                     cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    num_features = keypoints.size();
    return image_with_keypoints;
}

void FeatureDetector::setThreshold(int threshold) {
    this->threshold = threshold;
    fast_detector->setThreshold(threshold);
}

int FeatureDetector::getThreshold() const {
    return threshold;
}