#ifndef BUMPERDETECTION_KALMANFILTER_H
#define BUMPERDETECTION_KALMANFILTER_H

#include <opencv2/video/tracking.hpp>

#include "core/config/config_extraction.h"

class KalmanFilter {
public:

    KalmanFilter();

    cv::Vec3d update(double x, double y, double z, double dt);
private:
    void updateKalmanFilter();

    cv::KalmanFilter kf;

    double deltaTime;

    bool initialized = false;
};

#endif
