#ifndef BUMPERDETECTION_DETECTOR_H
#define BUMPERDETECTION_DETECTOR_H
#include "core/detection/structure/detection_data.h"
#ifdef __aarch64__
#define _CRT_SECURE_NO_WARNINGS
#include "rknn_api.h"
#endif

class Detector {
public:
    Detector();
    ~Detector();

    void initializeDetector();
    [[nodiscard]] std::vector<Detection> detect(const cv::Mat &img);

private:
#ifdef __aarch64__
    rknn_context ctx{}{};
#else
    cv::dnn::Net net;
#endif

    bool initialized;

    int INPUT_HEIGHT, INPUT_WIDTH;
    float NMS_THRESHOLD, CONF_THRESHOLD;

};


#endif //BUMPERDETECTION_DETECTOR_H
