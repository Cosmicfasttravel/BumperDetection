#ifndef BUMPERDETECTION_GET_MEASUREMENTS_H
#define BUMPERDETECTION_GET_MEASUREMENTS_H
#include <opencv2/core/types.hpp>
#include "core/detection/structure/detection_data.h"

namespace measurements {
    Position3D getXYZ(const Detection &detection, double measured_height);
    double getHeight(const cv::Mat &hsv, const Detection &detection);
    Color calculateColor(const cv::Mat &hsv, const Detection &detection);
}


#endif //BUMPERDETECTION_GET_MEASUREMENTS_H
