#include "kalman_filter.h"
#include "config_extraction.h"
#include <opencv2/core/mat.hpp>

kalmanFilter::kalmanFilter()
{
    kf.init(6, 3, 0, CV_64F);

    static const double dt = 1.0 / std::stoi(extractByTag("<avg_fps>"));

    kf.transitionMatrix = (cv::Mat_<double>(6,6) <<
        1, 0, 0, dt, 0,  0,
        0, 1, 0, 0,  dt, 0,
        0, 0, 1, 0,  0,  dt,
        0, 0, 0, 1,  0,  0,
        0, 0, 0, 0,  1,  0,
        0, 0, 0, 0,  0,  1
    );

    kf.measurementMatrix = (cv::Mat_<double>(3,6) <<
        1,0,0,0,0,0,
        0,1,0,0,0,0,
        0,0,1,0,0,0
    );

    double processNoise = std::stod(extractByTag("<process_noise>"));
    double measurementNoise = std::stod(extractByTag("<measurement_noise>"));
    double error = std::stod(extractByTag("<error>"));
    cv::setIdentity(kf.processNoiseCov, cv::Scalar(processNoise));//motion
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(measurementNoise));//noise
    cv::setIdentity(kf.errorCovPost, cv::Scalar(error));//measurement variance
}

cv::Vec3d kalmanFilter::update(double x, double y, double z, double dt)
{
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
