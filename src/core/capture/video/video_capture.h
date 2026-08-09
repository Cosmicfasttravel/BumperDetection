#ifndef BUMPERDETECTION_VIDEO_CAPTURE_H
#define BUMPERDETECTION_VIDEO_CAPTURE_H
#include "opencv2/opencv.hpp"
#include <queue>

class VideoCapture {
public:

    VideoCapture();
    ~VideoCapture();

    void writeToFile();
    void initializeVideoCapture();
    void shutdownVideoCapture();

    static void pushFrame(const cv::Mat &frame);

private:
    bool running = true;
    bool initialized = false;

    static std::mutex videoMutex;
    static std::condition_variable frameCv;

    int codec = {};

    std::thread vidThread;
    cv::VideoWriter writer;

    static std::queue<cv::Mat> frameQueue;

    void configureVideoCapture();

};


#endif //BUMPERDETECTION_VIDEO_CAPTURE_H
