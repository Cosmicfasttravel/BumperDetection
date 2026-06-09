#include "core/kalman/kalman_filter.h"
#include "core/config/config_extraction.h"

#include <opencv2/core/mat.hpp>

kalmanFilter::kalmanFilter() : k_config(), deltaTime(0) {
    kf.init(6, 3, 0, CV_64F);

    kf.transitionMatrix = (cv::Mat_<double>(6, 6) <<
                           1, 0, 0, deltaTime, 0, 0,
                           0, 1, 0, 0, deltaTime, 0,
                           0, 0, 1, 0, 0, deltaTime,
                           0, 0, 0, 1, 0, 0,
                           0, 0, 0, 0, 1, 0,
                           0, 0, 0, 0, 0, 1
    );

    kf.measurementMatrix = (cv::Mat_<double>(3, 6) <<
                            1, 0, 0, 0, 0, 0,
                            0, 1, 0, 0, 0, 0,
                            0, 0, 1, 0, 0, 0
    );
}

void kalmanFilter::init() {
    if (config::checkConfigVersion(k_config)) {
        k_config = config::getLatestCopy();

        cv::setIdentity(kf.processNoiseCov, cv::Scalar(k_config.position_kalman.process_noise)); //motion
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(k_config.position_kalman.measurement_noise)); //noise
        cv::setIdentity(kf.errorCovPost, cv::Scalar(k_config.position_kalman.error)); //measurement variance
    }
}

cv::Vec3d kalmanFilter::update(double x, double y, double z, double dt)
{
    init();

    deltaTime = dt;

    if (!initialized) {
        kf.statePost.at<double>(0) = x;
        kf.statePost.at<double>(1) = y;
        kf.statePost.at<double>(2) = z;
        kf.statePost.at<double>(3) = 0.0;
        kf.statePost.at<double>(4) = 0.0;
        kf.statePost.at<double>(5) = 0.0;

        initialized = true;
        return {x, y, z};
    }

    kf.transitionMatrix.at<double>(0,3) = dt;
    kf.transitionMatrix.at<double>(1,4) = dt;
    kf.transitionMatrix.at<double>(2,5) = dt;

    kf.predict();

    cv::Mat meas(3, 1, CV_64F);
    meas.at<double>(0) = x;
    meas.at<double>(1) = y;
    meas.at<double>(2) = z;

    cv::Mat est = kf.correct(meas);

    return {
        est.at<double>(0),
        est.at<double>(1),
        est.at<double>(2)
    };
}
