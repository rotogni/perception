#ifndef FEATURE_DETECTOR_HPP
#define FEATURE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <string>

class FeatureDetector {
public:
    // Constructor
    FeatureDetector(int threshold = 40);
    
    // Process a single image and return the visualization
    cv::Mat processImage(const cv::Mat& image, int& num_features);
    
    // Getters and setters
    void setThreshold(int threshold);
    int getThreshold() const;

private:
    cv::Ptr<cv::FastFeatureDetector> fast_detector;
    int threshold;
};

#endif 