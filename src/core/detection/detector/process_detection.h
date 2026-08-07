#ifndef BUMPERDETECTION_PROCESS_DETECTION_H
#define BUMPERDETECTION_PROCESS_DETECTION_H
#include "core/detection/structure/detection_data.h"


namespace process_detection {
    std::vector<Detection> ProcessYoloOutput(
    const std::vector<cv::Mat> &outputs,
    int img_width,
    int img_height,
    int input_width,
    int input_height,
    float conf_threshold,
    float nms_threshold);
}

#endif //BUMPERDETECTION_PROCESS_DETECTION_H
