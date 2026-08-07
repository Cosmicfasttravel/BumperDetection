#ifndef BUMPERDETECTION_GET_MEASUREMENTS_H
#define BUMPERDETECTION_GET_MEASUREMENTS_H
#include <opencv2/core/types.hpp>
#include "core/detection/structure/detection_data.h"

namespace measurements {
    Position3D getXYZ();
    double getHeight(const cv::Mat &hsv, const cv::Rect &box);
}


#endif //BUMPERDETECTION_GET_MEASUREMENTS_H
