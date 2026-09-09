#ifndef BUMPERDETECTION_DETECTION_DATA_H
#define BUMPERDETECTION_DETECTION_DATA_H
#include <opencv2/opencv.hpp>
#include "global/enums.h"

struct Position3D {
    double x;
    double y;
    double z;
};

struct Detection{
    cv::Rect boundingBox;

    Color color;

    int teamNumber;
    double confidence;

    Position3D pos;

    bool noise;
};

#endif //BUMPERDETECTION_DETECTION_DATA_H
