#include "analyze_detections.h"
#include "debug_log.h"
#include <deque>
#include "config_extraction.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <sstream>
#include <fstream>
#include <ostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <semaphore>
#include <climits>
#include <string>
#include <tesseract/baseapi.h>
#include <chrono>
#include <vector>
#include <leptonica/allheaders.h>
#include <filesystem>
#include <future>
#include <iostream>
#include <unordered_set>
#include <opencv2/video/tracking.hpp>
#include "kalman_filter.h"
#include "thread_manager.h"

std::atomic<int> ocrCounter{0};
std::atomic<bool> cleanUp = false;


struct TrackedRobot {
    int x = -1;
    int y = -1;
    int lostCounter = 0;

    std::string robot_id = "-1";
    std::string teamNumber;

    bool used = false;

    std::chrono::steady_clock::time_point timestamp;

    int ocrCount = 0;
};

struct OutputData {
    double x{};
    double y{};
    double z{};
    std::string label;
    Detection det;
    int id{};
};

std::vector<TrackedRobot> tracked;
std::vector<std::string> visibleIDs;

int levenshteinDist(const std::string &word1, const std::string &word2) {
    const int size1 = static_cast<int>(word1.size());
    const int size2 = static_cast<int>(word2.size());
    std::vector verif(size1 + 1, std::vector<int>(size2 + 1));

    if (size1 == 0)
        return size2;
    if (size2 == 0)
        return size1;

    for (int i = 0; i <= size1; i++)
        verif[i][0] = i;
    for (int j = 0; j <= size2; j++)
        verif[0][j] = j;

    for (int i = 1; i <= size1; i++) {
        for (int j = 1; j <= size2; j++) {
            int cost = (word2[j - 1] == word1[i - 1]) ? 0 : 1;
            verif[i][j] = std::min(
                std::min(verif[i - 1][j] + 1, verif[i][j - 1] + 1),
                verif[i - 1][j - 1] + cost);
        }
    }

    return verif[size1][size2];
}

double getDistance(const double height, const Config &config) {
    double focal_length_cm = config.camera.focal_length;
    double known_height_cm = config.bumper.height;
    double pixel_height_cm = config.camera.pixel_height;

    return (height > 0) ? (known_height_cm * focal_length_cm) / (height * pixel_height_cm) : 0.0;
}

std::mutex Filter_Mutex;
static std::unordered_map<std::string, kalmanFilter> filters;

std::vector<double> getMeasurements(double distance, const Detection &detection, const Config &config, double dt) {
    if (dt <= 0.0 || dt > 1.0) dt = 0.0033;

    std::string id = detection.id;

    double SCREEN_WIDTH = config.screen.width;
    double SCREEN_HEIGHT = config.screen.height;
    double X_FOV = config.screen.x_fov;
    double Y_FOV = config.screen.y_fov;

    double max_cord_x = SCREEN_WIDTH / 2;
    double max_cord_y = SCREEN_HEIGHT / 2;

    double abs_bounding_x = detection.bounding_box.x + (0.5 * detection.bounding_box.width); // middle x
    double abs_bounding_y = detection.bounding_box.y + (detection.bounding_box.height); // bottom y

    double offset_x = (abs_bounding_x - max_cord_x) / max_cord_x;
    double offset_y = (abs_bounding_y - max_cord_y) / max_cord_y;

    const double x_angle = (X_FOV / 2.0 * offset_x) * CV_PI / 180.0;
    const double y_angle = (Y_FOV / 2.0 * offset_y) * CV_PI / 180.0;

    const double x_coordinate = (distance / 100.0) * cos(y_angle) * cos(x_angle);
    const double y_coordinate = (distance / 100.0) * sin(x_angle);
    const double z_coordinate = (distance / 100.0) * sin(y_angle) * cos(x_angle);

    cv::Vec3d filtered;
    filtered[0] = x_coordinate;
    filtered[1] = y_coordinate;
    filtered[2] = z_coordinate;

    Filter_Mutex.lock();
    if (!id.empty()) {
        auto it = filters.find(id);
        if (it == filters.end()) {
            it = filters.emplace(id, kalmanFilter(config.position_kalman.process_noise,
                                                  config.position_kalman.measurement_noise,
                                                  config.position_kalman.error)).first;
        }
        kalmanFilter &filter = it->second;
        filtered = filter.update(x_coordinate, y_coordinate, z_coordinate, dt);
    }

    for (auto it = filters.begin(); it != filters.end();) {
        bool found = false;
        for (auto &num: visibleIDs) {
            if (it->first == num) {
                found = true;
            }
        }
        if (!found) {
            it = filters.erase(it);
        } else {
            ++it;
        }
    }
    Filter_Mutex.unlock();

    return {filtered[0], filtered[1], filtered[2]};
}

std::string getRobotLabel(Detection &det, const cv::Mat &hsv, const Config &config) {
    try {
        auto maxOCR = config.ocr.max_instances;
        if (ocrCounter >= maxOCR)
            return "";
        ++ocrCounter;

        if (det.color.empty())
            return "";

        static thread_local std::unique_ptr<tesseract::TessBaseAPI> api;
        static thread_local bool init = false;

        if (cleanUp) {
            if (!api)
                return "-1";

            api->End();
            api.reset();
            return "-1";
        }

        if (!init) {
            api = std::make_unique<tesseract::TessBaseAPI>();
            if (config.ocr.mode == "default" || config.ocr.mode == "tessonly") api->Init(
                config.ocr.tessdata_path.c_str(), "eng", tesseract::OEM_TESSERACT_ONLY);
            if (config.ocr.mode == "lstmonly") api->Init(config.ocr.tessdata_path.c_str(), "eng",
                                                         tesseract::OEM_LSTM_ONLY);
            if (config.ocr.mode == "combined") api->Init(config.ocr.tessdata_path.c_str(), "eng",
                                                         tesseract::OEM_TESSERACT_LSTM_COMBINED);

            api->SetPageSegMode(tesseract::PSM_SINGLE_WORD);
            api->SetVariable("tessedit_char_whitelist", "0123456789");
            init = true;
        }

        cv::Rect safeBB = det.bounding_box & cv::Rect(0, 0, hsv.cols, hsv.rows);
        if (safeBB.empty()) { --ocrCounter; return ""; }
        cv::Mat img = hsv(safeBB).clone();

        cv::Mat colorMask;

        cv::inRange(
            img, cv::Scalar(config.ocr.mask_thresholds.hue_lower, config.ocr.mask_thresholds.saturation_lower,
                            config.ocr.mask_thresholds.value_lower),
            cv::Scalar(config.ocr.mask_thresholds.hue_upper, config.ocr.mask_thresholds.saturation_upper,
                       config.ocr.mask_thresholds.value_upper), colorMask);

        if (colorMask.cols < config.ocr.min_img_size) {
            double scale = static_cast<float>(config.ocr.min_img_size) / static_cast<float>(colorMask.cols);
            cv::resize(colorMask, colorMask, cv::Size(), scale, scale, cv::INTER_CUBIC);
        }

        cv::Mat final;
        cv::bitwise_not(colorMask, final);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                                   cv::Size(config.ocr.morphology_kernel_size,
                                                            config.ocr.morphology_kernel_size));
        cv::morphologyEx(final, final, cv::MORPH_OPEN, kernel);

        if (final.empty() || final.cols <= 0 || final.rows <= 0) {
            --ocrCounter;
            return "";
        }

        api->SetImage(final.data, final.cols, final.rows, 1, final.step);
        api->SetSourceResolution(70);
        char *outText = api->GetUTF8Text();
        std::string result(outText);
        delete[] outText;

        std::erase_if(result, ::isspace);

        int minIndex = 0;

        int minDist = INT_MAX;
        if (!result.empty() && std::ranges::all_of(result, ::isdigit)) {
            for (int i = 0; i < 5; i++) {
                int d = {};

                if (det.color == "blue") d = levenshteinDist(result, config.teams.blueTeams[i]);
                if (det.color == "red") d = levenshteinDist(result, config.teams.redTeams[i]);

                if (d < minDist) {
                    minDist = d;
                    minIndex = i;
                }
            }
        }

        if (det.color == "blue") result = config.teams.blueTeams[minIndex];
        if (det.color == "red") result = config.teams.redTeams[minIndex];

        if (minDist > config.ocr.lev_distance) {
            result = "";
        }

        return result;
    } catch (std::exception &e) {
        log("Tesseract Failure, " + std::string(e.what()), spdlog::level::critical);
    } catch (...) {
        log("Unknown Tesseract Failure", spdlog::level::critical);
    }
    return "";
}

OutputData analyzeDetection(
    const cv::Mat &hsv,
    Detection det,
    const Config &config,
    double dt) {
    if (det.bounding_box.x < 0 || det.bounding_box.y < 0 || det.bounding_box.x + det.bounding_box.width > config.screen.
        width || det.bounding_box.y + det.bounding_box.height > config.screen.height) {
    }
    if (hsv.empty()) return OutputData{};

    auto bumperBoundingBox = hsv(det.bounding_box).clone();

    auto centerX = det.bounding_box.x + (0.5 * det.bounding_box.width);
    auto centerY = det.bounding_box.y + (0.5 * det.bounding_box.height);

    int relCenterX = static_cast<int>(centerX) - det.bounding_box.x;
    int relCenterY = static_cast<int>(centerY) - det.bounding_box.y;

    //Red thresholds
    auto lowerRedThreshold_1 = cv::Scalar(config.height_measurement.red_mask_thresholds_1.hue_lower,
                                          config.height_measurement.red_mask_thresholds_1.saturation_lower,
                                          config.height_measurement.red_mask_thresholds_1.value_lower);
    auto upperRedThreshold_1 = cv::Scalar(config.height_measurement.red_mask_thresholds_1.hue_upper,
                                          config.height_measurement.red_mask_thresholds_1.saturation_upper,
                                          config.height_measurement.red_mask_thresholds_1.value_upper);

    auto lowerRedThreshold_2 = cv::Scalar(config.height_measurement.red_mask_thresholds_2.hue_lower,
                                          config.height_measurement.red_mask_thresholds_2.saturation_lower,
                                          config.height_measurement.red_mask_thresholds_2.value_lower);
    auto upperRedThreshold_2 = cv::Scalar(config.height_measurement.red_mask_thresholds_2.hue_upper,
                                          config.height_measurement.red_mask_thresholds_2.saturation_upper,
                                          config.height_measurement.red_mask_thresholds_2.value_upper);

    //Blue threshold
    auto lowerBlueThreshold = cv::Scalar(config.height_measurement.blue_mask_thresholds.hue_lower,
                                         config.height_measurement.blue_mask_thresholds.saturation_lower,
                                         config.height_measurement.blue_mask_thresholds.value_lower);
    auto upperBlueThreshold = cv::Scalar(config.height_measurement.blue_mask_thresholds.hue_upper,
                                         config.height_measurement.blue_mask_thresholds.saturation_upper,
                                         config.height_measurement.blue_mask_thresholds.value_upper);

    cv::Mat rMask1, rMask2, bMask, finalMask, contours;
    cv::inRange(bumperBoundingBox, lowerRedThreshold_1, upperRedThreshold_1, rMask1);
    cv::inRange(bumperBoundingBox, lowerRedThreshold_2, upperRedThreshold_2, rMask2);
    cv::bitwise_or(rMask1, rMask2, rMask1);

    cv::inRange(bumperBoundingBox, lowerBlueThreshold, upperBlueThreshold, bMask);

    relCenterX = std::clamp(relCenterX, 0, rMask1.cols - 1);
    relCenterY = std::clamp(relCenterY, 0, rMask1.rows - 1);

    if (rMask1.at<int>(relCenterY, relCenterX) > 0) det.color = "red";
    else if (bMask.at<int>(relCenterY, relCenterX) > 0) det.color = "blue";
    else det.color = "";

    for (const auto &t: tracked) {
        if (t.robot_id == det.id) {
            if (t.teamNumber.empty() && t.ocrCount < 6) {
                det.teamNumber = getRobotLabel(det, hsv, config);
            }
        }
    }

    cv::bitwise_or(rMask1, bMask, finalMask);

    double height = 0;

    auto topY = det.bounding_box.y;
    auto bottomY = det.bounding_box.y + det.bounding_box.height;

    for (auto x = det.bounding_box.x; x < det.bounding_box.x + det.bounding_box.width; x++) {
        int relX = std::clamp(x - det.bounding_box.x, 0, finalMask.cols - 1);
       
        height = 0;
        for (auto y = topY; y < bottomY; y++) {
            int relY = std::clamp(y - det.bounding_box.y, 0, finalMask.rows - 1);
            int color = finalMask.at<int>(y, centerX);
            if (color > 0) height++;
        }
    }

    std::vector<double> measurements = getMeasurements(getDistance(height, config), det, config, dt);

    OutputData data;
    data.x = measurements[0], data.y = measurements[1], data.z = measurements[2], data.label = det.teamNumber, data.det
            = det;

    return data;
}

static std::unique_ptr<ThreadManager> thread_manager;

void detectionScheduler(cv::Mat &frame, std::vector<Detection> &detections, const Config &config) {
    tracked.reserve(5);

    if (detections.empty()) return;

    if (!thread_manager) thread_manager = std::make_unique<ThreadManager>(config.thread_pool_size);

    visibleIDs.clear();

    static std::deque<int> availableIDs;
    static bool idFilled = false;
    if (!idFilled) {
        for (int i = 0; i < config.tracking.id_count; i++) availableIDs.push_back(i);
        idFilled = true;
    }

    for (auto &t: tracked) t.used = false;

    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);


    //switch away from greedy tracking
    for (auto &det: detections) {
        int centerX = det.bounding_box.x + det.bounding_box.width / 2;
        int centerY = det.bounding_box.y + det.bounding_box.height / 2;

        TrackedRobot *bestMatch = nullptr;
        double minDist = std::numeric_limits<double>::max();

        for (auto &t: tracked) {
            if (t.used)

                continue;

            double dx = centerX - t.x;
            double dy = centerY - t.y;
            double dist = std::sqrt(dx * dx + dy * dy);

            if (dist < minDist && dist < config.tracking.max_distance_threshold_x) {
                minDist = dist;
                bestMatch = &t;
            }
        }

        if (bestMatch) {
            bestMatch->x = centerX;
            bestMatch->y = centerY;
            bestMatch->lostCounter = 0;
            bestMatch->used = true;

            det.id = bestMatch->robot_id;
            if (!det.teamNumber.empty())
                bestMatch->teamNumber = det.teamNumber;

            std::stringstream ss;
            ss << "ID: " << det.id << " " << "Label: " << bestMatch->teamNumber;
            cv::putText(frame, ss.str(), cv::Point(det.bounding_box.x, det.bounding_box.y), cv::FONT_HERSHEY_SIMPLEX,
                        0.7, cv::Scalar(255, 255, 255), 2);

            det.meta.dt = std::chrono::duration<double>(det.timestamp - bestMatch->timestamp).count();
            bestMatch->ocrCount = 0;

            bestMatch->timestamp = det.timestamp;
        } else if (!availableIDs.empty()) {
            int newID = availableIDs.front();
            availableIDs.pop_front();

            TrackedRobot newRobot;
            newRobot.x = centerX;
            newRobot.y = centerY;
            newRobot.robot_id = std::to_string(newID);
            newRobot.lostCounter = 0;
            newRobot.used = true;
            newRobot.timestamp = det.timestamp;
            newRobot.ocrCount = 0;

            tracked.push_back(newRobot);
            det.id = newRobot.robot_id;
        } else {
            det.id.clear();
        }
    }

    for (auto it = tracked.begin(); it != tracked.end();) {
        if (!it->used)
            it->lostCounter++;

        if (it->lostCounter >= config.tracking.lost_threshold) {
            if (it->robot_id != "-1") {
                availableIDs.push_back(std::stoi(it->robot_id));
            }
            it = tracked.erase(it);
        } else
            ++it;
    }
    for (const auto &det: detections) {
        visibleIDs.emplace_back(det.id);
    }
    std::vector<OutputData> results = {};

    std::vector<std::future<OutputData> > futures;
    ocrCounter = 0;
    for (const auto &detection: detections) {
        futures.push_back(thread_manager->enqueue([hsv, config, detection]() {
            try {
                return analyzeDetection(hsv, detection, config, detection.meta.dt);
            } catch (...) {
                log("Problem occurred with thread scheduling", spdlog::level::warn);
                return OutputData{};
            }
        }));
    }
    results.reserve(futures.size());

    for (auto &fut: futures) {
        results.push_back(fut.get());
    }

    for (size_t i = 0; i < detections.size(); i++) {
        detections[i].teamNumber = results[i].label;
    }

    for (auto &det: detections) {
        for (auto &result: results) {
            if (result.det.color.empty())
                cv::line(frame, cv::Point(640, 720),
                         cv::Point(result.det.bounding_box.x + result.det.bounding_box.width / 2,
                                   result.det.bounding_box.y + result.det.bounding_box.height / 2),
                         cv::Scalar(255, 255, 255), 1);
            else
                cv::line(frame, cv::Point(640, 720),
                         cv::Point(result.det.bounding_box.x + result.det.bounding_box.width / 2,
                                   result.det.bounding_box.y + result.det.bounding_box.height / 2),
                         result.det.color == "blue" ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255), 1);
        }

        for (auto &t: tracked) {
            if (det.id == t.robot_id && !det.teamNumber.empty()) {
                t.teamNumber = det.teamNumber;
            }
        }
    }

    for (auto &result: results) {
        std::stringstream ss;
        ss << "X: " << result.x << " " << "Y: " << result.y << " Color: " << result.det.color;
        cv::putText(frame, ss.str(), cv::Point(result.det.bounding_box.x, result.det.bounding_box.y - 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }
}

void clean(const Config &config) {
    cleanUp = true; {
        Detection det = {};
        cv::Mat hsv = {};

        std::vector<std::future<std::string> > futures;

        for (int i = 0; i < config.thread_pool_size; ++i) {
            futures.push_back(thread_manager->enqueue([hsv, config, &det]() {
                try {
                    return getRobotLabel(det, hsv, config);
                } catch (...) {
                    log("Problem occurred with clean up", spdlog::level::err);
                    std::string s;
                    return s;
                }
            }));
        }
        for (auto &fut: futures) {
            fut.get();
        }
        thread_manager->shutdown();
    }
}
