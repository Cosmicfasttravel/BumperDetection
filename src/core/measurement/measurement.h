#ifndef BUMPERDETECTION_GET_MEASUREMENTS_H
#define BUMPERDETECTION_GET_MEASUREMENTS_H
#include <opencv2/core/types.hpp>

namespace measurements {
    struct Position3D {
        double x;
        double y;
        double z;
    };

    Position3D getXYZ();
    double getHeight(const cv::Mat &hsv, const cv::Rect &box);
}


#endif //BUMPERDETECTION_GET_MEASUREMENTS_H
