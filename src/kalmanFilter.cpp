#include "kalmanFilter.h"
#include <opencv2/core/mat.hpp>

kalmanFilter::kalmanFilter()
{
    kf.init(6, 3, 0, CV_64F);

    double dt = 1.0 / 5.0; // change to 1/ AVG fps

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

    cv::setIdentity(kf.processNoiseCov, cv::Scalar(100));//motion
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(100));//noise
    cv::setIdentity(kf.errorCovPost, cv::Scalar(200));//measurement variance
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