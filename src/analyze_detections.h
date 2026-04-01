#ifndef BUMPERDETECTION_EDGEDETECTION_H
#define BUMPERDETECTION_EDGEDETECTION_H
#include <opencv2/core/mat.hpp>
#include <tesseract/baseapi.h>
#include "config_extraction.h"

struct Detection {
    int class_id;
    float confidence;
    cv::Rect bounding_box;
    
    std::string class_name;
    std::string id;
    std::string color;
    std::string label;

    bool tracked;
};

void analyzeDetections(
    const std::string teamNumbers[5],
    cv::Mat &frame,
    std::vector<Detection> &detections, 
    const Config& config
);

inline tesseract::TessBaseAPI *api;
void startOCR();
void endOCR();

#endif //BUMPERDETECTION_EDGEDETECTION_H
