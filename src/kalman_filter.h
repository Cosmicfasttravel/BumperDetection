#ifndef BUMPERDETECTION_KALMANFILTER_H
#define BUMPERDETECTION_KALMANFILTER_H
#include <opencv2/video/tracking.hpp>
#include "config_extraction.h"
class kalmanFilter {
public:
    kalmanFilter(const Config& config);
    cv::Vec3d update(double x, double y, double z, double dt);

private:
    cv::KalmanFilter kf;
    bool initialized = false;
};


#endif
