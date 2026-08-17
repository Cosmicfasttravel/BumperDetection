#include <future>

#include "core/detection/detector/detector.h"
#include "core/capture/capture.h"
#include "core/kalman/kalman_filter.h"
#include "core/measurement/measurement.h"
#include "tesseract/tesseract_engine.h"
#include "core/threading/thread_manager.h"
#include "log/logger.h"

using Clock = std::chrono::high_resolution_clock;

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

    std::vector<double> fps;
    auto prev_frame_time = Clock::now();

    while (true) {
        std::stringstream ss;

        auto frame_start = Clock::now();
        prev_frame_time = frame_start;

        config::tryUpdate();

        cv::Mat frame = cameraCapture.retrieveLatestFrame();

        if (frame.empty()) continue;

        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        std::vector<Detection> detections;
        detections = detector.detect(frame);

        //Untrimmed bounding boxes
        for (const auto &detection: detections) {
            cv::rectangle(frame, cv::Point(detection.boundingBox.x, detection.boundingBox.y),
                          cv::Point(detection.boundingBox.x + detection.boundingBox.width,
                                    detection.boundingBox.y + detection.boundingBox.height),
                          cv::Scalar(255, 255, 255));
        }

        for (auto &detection: detections) {
            detection.color = measurements::calculateColor(hsv, detection);
            cv::Rect trim = measurements::trimBoundingBox(hsv, detection);

            if (trim.empty()) {
                detection.noise = true;
                continue;
            }
            detection.boundingBox = trim;
        }

        //enqueue in a thread
        for (auto &detection: detections) {
            double height = measurements::getHeight(hsv, detection);
            detection.pos = measurements::getXYZ(detection, height);
        }

        std::vector<std::future<std::string> > futures;
        std::vector<std::string> results = {};
        for (const auto &detection: detections) {
            if (detection.noise) continue;
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
        }

        //Trimmed Bounding boxes
        for (const auto &detection: detections) {
            if (detection.noise) continue;
            cv::rectangle(frame, cv::Point(detection.boundingBox.x, detection.boundingBox.y),
                          cv::Point(detection.boundingBox.x + detection.boundingBox.width,
                                    detection.boundingBox.y + detection.boundingBox.height),
                          cv::Scalar(255, 0, 255));

            std::stringstream stringStream;
            stringStream << "X: " << detection.pos.x << " " << "Y: " << detection.pos.y << " Color: " << ((detection.color == 0) ? "RED" : (detection.color == 1) ? "BLUE" : "NONE");
            cv::putText(frame, stringStream.str(), cv::Point(detection.boundingBox.x, detection.boundingBox.y - 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        }

        auto frame_end = Clock::now();
        using FrameDuration = std::chrono::duration<double>;
        auto delta = FrameDuration(frame_end - prev_frame_time).count();

        ss.clear();
        double sum = 0;
        for (double fp : fps)
        {
            sum += fp;
        }

        ss << std::fixed << std::setprecision(2) << "FPS: " << ((fps.empty()) ? 0 : (sum / static_cast<double>(fps.size())));
        fps.emplace_back((1.f / delta));

        if (fps.size() >= 20)
        {
            fps.erase(fps.begin());
        }

        cv::putText(frame, ss.str(), cv::Point(10, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 255), 2);

        cv::imshow("", frame);

        int key = cv::waitKey(1);
        if (key == 27) break;
    }

    return 0;
}
