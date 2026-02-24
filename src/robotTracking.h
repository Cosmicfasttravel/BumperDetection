#ifndef BUMPERDETECTION_EDGEDETECTION_H
#define BUMPERDETECTION_EDGEDETECTION_H
#include <opencv2/core/mat.hpp>
#include <tesseract/baseapi.h>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect bounding_box;
    std::string class_name;
    std::string label;
    std::vector<std::vector<cv::Point>>::value_type largestContour {};
    int largestContourSize {};
    cv::Scalar largestContourColor {};
};


void analyzeDetections(
    cv::Mat &blankFrame,
    const std::string teamNumbers[5],
    cv::Mat &frame,
    std::vector<Detection> &detections
);

inline auto *api = new tesseract::TessBaseAPI();
void startOCR();
void endOCR();

#endif //BUMPERDETECTION_EDGEDETECTION_H
