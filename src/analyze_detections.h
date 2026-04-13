#ifndef BUMPERDETECTION_EDGEDETECTION_H
#define BUMPERDETECTION_EDGEDETECTION_H
#include <opencv2/core/mat.hpp>
#include <tesseract/baseapi.h>
#include "config_extraction.h"

struct Detection {
    float confidence;
    cv::Rect bounding_box;
    std::string color, id, teamNumber;
};

void detectionScheduler(
    cv::Mat &frame,
    std::vector<Detection> &detections, 
    const Config& config
);
void cleanUp();
#endif //BUMPERDETECTION_EDGEDETECTION_H
