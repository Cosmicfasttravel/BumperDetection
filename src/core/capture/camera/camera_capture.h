#ifndef BUMPERDETECTION_CAMERA_CAPTURE_H
#define BUMPERDETECTION_CAMERA_CAPTURE_H
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

class CameraCapture {
public:
    CameraCapture();
    ~CameraCapture();

    void initializeCameraCapture();
    cv::Mat retrieveLatestFrame();
    void configureCaptureComponent();
    void shutdownCaptureComponent();
    void runtimeConfigure();

private:
    bool shutdown = true;
    bool initialized = false;
    bool videoMode = false;

    std::mutex frameMutex;
    std::atomic<bool> capturing;

    cv::Mat currentFrame;
    std::thread camThread;

    cv::VideoCapture cap;

    void capture(cv::VideoCapture &capture);

};

#endif //BUMPERDETECTION_CAMERA_CAPTURE_H
