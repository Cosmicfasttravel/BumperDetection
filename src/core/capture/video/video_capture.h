#ifndef BUMPERDETECTION_VIDEO_CAPTURE_H
#define BUMPERDETECTION_VIDEO_CAPTURE_H
#include "opencv2/opencv.hpp"
#include <queue>

class video_capture {
public:

    video_capture();
    ~video_capture();

    void writeToFile();
    void initializeVideoCapture();
    void shutdownVideoCapture();

    static void pushFrame(const cv::Mat &frame);

private:
    bool captureMode = true;
    bool running = true;

    static std::mutex videoMutex;
    static std::condition_variable frameCv;

    int codec = {};

    std::thread vidThread;
    cv::VideoWriter writer;

    static std::queue<cv::Mat> frameQueue;

    void configureVideoCapture();

};


#endif //BUMPERDETECTION_VIDEO_CAPTURE_H
