#include "core/detection/detector/detector.h"
#include "core/capture/capture.h"
#include "core/kalman/kalman_filter.h"
#include "core/measurement/measurement.h"

int main() {
    config::tryUpdate();

    Detector detector;
    detector.initializeDetector();

    bool videoMode = false;

    VideoCapture videoCapture;
    if (videoMode) {
        videoCapture.initializeVideoCapture();
        videoCapture.writeToFile();
    }

    CameraCapture cameraCapture;
    cameraCapture.initializeCameraCapture();

    KalmanFilter kf;

    while (true) {
        config::tryUpdate();

        cv::Mat frame = cameraCapture.retrieveLatestFrame();

        if (frame.empty()) continue;

        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        std::vector<Detection> detections;
        detections = detector.detect(frame);

        //trim bounding boxes here

        //enqueue in a thread
        for (auto& detection : detections) {
            double height = measurements::getHeight(hsv, detection);
            detection.pos = measurements::getXYZ(detection, height);
        }

        //tess engine in a different thread

        for (const auto& detection : detections) {
            cv::rectangle(frame, cv::Point(detection.boundingBox.x, detection.boundingBox.y),
                cv::Point(detection.boundingBox.x + detection.boundingBox.width, detection.boundingBox.y + detection.boundingBox.height),
                cv::Scalar(255, 255, 255));
        }

        cv::imshow("", frame);

        int key = cv::waitKey(1);
        if (key == 27) break;
    }

    return 0;
}

