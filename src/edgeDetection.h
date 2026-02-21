#ifndef BUMPERDETECTION_EDGEDETECTION_H
#define BUMPERDETECTION_EDGEDETECTION_H
#include <list>
#include <opencv2/core/mat.hpp>
#include <tesseract/baseapi.h>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect bounding_box;
    std::string class_name;
    std::vector<std::string> label_list;
};


void detectEdgesBumper(
    cv::Mat &blankFrame,
    const std::string teamNumbers[5],
    cv::Mat &frame,
    std::vector<Detection> &detections
);

inline auto *api = new tesseract::TessBaseAPI();
void startOCR();
void endOCR();

#endif //BUMPERDETECTION_EDGEDETECTION_H
