#ifndef BUMPERDETECTION_EDGEDETECTION_H
#define BUMPERDETECTION_EDGEDETECTION_H
#include <opencv2/core/mat.hpp>
#include <tesseract/baseapi.h>
#include "config_extraction.h"

struct TrackingMeta {
    double dt;
};

struct Detection {
    float confidence;
    cv::Rect bounding_box;
    std::string color, id, teamNumber;
    std::chrono::steady_clock::time_point timestamp;
    TrackingMeta meta;
};

void detectionScheduler(
    cv::Mat &frame,
    std::vector<Detection> &detections, 
    const Config& config
);
void cleanUp();
#endif //BUMPERDETECTION_EDGEDETECTION_H
