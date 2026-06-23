#ifndef BUMPERDETECTION_CAMERA_CAPTURE_H
#define BUMPERDETECTION_CAMERA_CAPTURE_H
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

class camera_capture {
public:
    camera_capture();
    ~camera_capture();

    void initializeCaptureComponent();
    cv::Mat retrieveLatestFrame();
    void shutdownCaptureComponent();

private:
    bool shutdown = true;
    bool videoMode = false;

    std::mutex frameMutex;
    std::atomic<bool> capturing;

    cv::Mat currentFrame;
    std::thread camThread;

    cv::VideoCapture cap;

    void capture(cv::VideoCapture &capture);
    void configureCaptureComponent();

};

#endif //BUMPERDETECTION_CAMERA_CAPTURE_H
