#ifndef BUMPERDETECTION_DETECTION_DATA_H
#define BUMPERDETECTION_DETECTION_DATA_H
#include <opencv2/opencv.hpp>
#include "global/enums.h"

struct Detection{
    cv::Rect boundingBox;

    Color color;

    int teamNumber;
    double confidence;
};

#endif //BUMPERDETECTION_DETECTION_DATA_H
