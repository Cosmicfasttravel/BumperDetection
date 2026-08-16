#include <future>

#include "core/detection/detector/detector.h"
#include "core/capture/capture.h"
#include "core/kalman/kalman_filter.h"
#include "core/measurement/measurement.h"
#include "tesseract/tesseract_engine.h"
#include "core/threading/thread_manager.h"
#include "log/logger.h"

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

    ThreadManager tessThreadManager(config::getLatestCopy().thread_pool_size);

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

            detection.color = measurements::calculateColor(hsv, detection);
        }

        std::vector<std::future<std::string>> futures;
        std::vector<std::string> results = {};
        for (const auto &detection : detections) {
            futures.push_back(tessThreadManager.enqueue([hsv, detection]() {
                try {
                    return TesseractEngine::current().tesseractEngine(hsv, detection);
                } catch (...) {
                    logging::write("Problem occurred with thread scheduling", spdlog::level::warn);
                    return std::string("");
                }
            }));
        }
        results.reserve(futures.size());

        for (size_t i = 0; i < futures.size(); ++i) {
            auto result = futures[i].get();
            if (!result.empty()) detections[i].teamNumber = std::stoi(result);
            std::cout << detections[i].teamNumber << std::endl;
        }

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

