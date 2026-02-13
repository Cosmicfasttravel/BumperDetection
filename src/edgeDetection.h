#ifndef BUMPERDETECTION_EDGEDETECTION_H
#define BUMPERDETECTION_EDGEDETECTION_H
#include <opencv2/core/mat.hpp>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect bounding_box;
    std::string class_name;
    std::string label;
};


class edgeDetection {
public:
private:
};

void detectEdgesBumper(
    cv::Mat &blankFrame,
    const std::string teamNumbers[5],
    cv::Mat &frame,
    std::vector<Detection> &detections
);



void findNumbers(std::vector<Detection> &detections, const cv::Mat &blankFrame, cv::Mat &frame,
                 const std::string teamNumbers[5]);

#endif //BUMPERDETECTION_EDGEDETECTION_H
