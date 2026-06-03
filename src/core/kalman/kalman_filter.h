#ifndef BUMPERDETECTION_KALMANFILTER_H
#define BUMPERDETECTION_KALMANFILTER_H
#include <opencv2/video/tracking.hpp>
#include "../config/config_extraction.h"

class kalmanFilter {
public:

    kalmanFilter();

    cv::Vec3d update(double x, double y, double z, double dt);
private:
    void init();

    cv::KalmanFilter kf;

    double deltaTime;

    bool initialized = false;
    uint64_t configVersion = 0;
};

#endif
